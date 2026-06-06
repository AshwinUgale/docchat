/**
 * ``DocChat: Set up sidecar`` command - v1.0.1.
 *
 * Why this exists:
 *   v1.0.0 shipped to the VS Code Marketplace as a TypeScript-only .vsix.
 *   The Python sidecar (`docchat_sidecar`) was assumed to already be on the
 *   user's system, which is true on the developer's box but false for
 *   anyone installing from the Marketplace fresh. The first time anyone
 *   ran `DocChat: Open Chat Panel` they got
 *
 *       DocChat: sidecar exited before port line (code 1)
 *
 *   because `python -m docchat_sidecar` failed with ModuleNotFoundError.
 *
 * What v1.0.1 does:
 *   - Bundles the sidecar Python source inside the .vsix at sidecar-source/
 *     (see extension/scripts/prepackage.mjs).
 *   - Adds this setup command that copies the bundled source into
 *     ~/.docchat/sidecar/, runs `uv sync` there, and writes the resulting
 *     venv's python path to the docchat.sidecarPython setting (global).
 *
 * Prerequisites the setup itself checks:
 *   - Python 3.11+ on PATH (or invocable via `py -3.11` on Windows).
 *   - uv on PATH (https://docs.astral.sh/uv/).
 *
 * Out of scope for v1.0.1 (deliberately, per ADR-014):
 *   - Auto-installing Python or uv. The setup prints a clear error with
 *     the install URL instead, so users know what to fetch.
 *   - Auto-running setup on first activate without user opt-in. The user
 *     gets prompted via the openPanel error path, then chooses Run Setup.
 */

import { spawnSync } from "node:child_process";
import { cpSync, existsSync, mkdirSync, rmSync } from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import * as vscode from "vscode";

const PYTHON_MIN_MAJOR = 3;
const PYTHON_MIN_MINOR = 11;

const PYTHON_INSTALL_URL = "https://www.python.org/downloads/";
const UV_INSTALL_URL = "https://docs.astral.sh/uv/getting-started/installation/";

interface ToolCheck {
  found: boolean;
  command: string;
  argv: string[];
  version?: string;
  message?: string;
}

/**
 * Try a list of candidate Python invocations and return the first one that
 * reports >= 3.11. On Windows we prefer the `py` launcher because it knows
 * about every installed Python regardless of PATH order.
 */
function checkPython(): ToolCheck {
  const candidates: Array<{ command: string; argv: string[] }> = process.platform === "win32"
    ? [
        { command: "py", argv: ["-3.12"] },
        { command: "py", argv: ["-3.11"] },
        { command: "py", argv: ["-3"] },
        { command: "python3", argv: [] },
        { command: "python", argv: [] },
      ]
    : [
        { command: "python3.12", argv: [] },
        { command: "python3.11", argv: [] },
        { command: "python3", argv: [] },
        { command: "python", argv: [] },
      ];
  for (const { command, argv } of candidates) {
    const res = spawnSync(command, [...argv, "--version"], { encoding: "utf-8" });
    if (res.status !== 0) continue;
    const out = (res.stdout || res.stderr || "").trim();
    const m = /Python\s+(\d+)\.(\d+)/.exec(out);
    if (!m) continue;
    const major = Number.parseInt(m[1]!, 10);
    const minor = Number.parseInt(m[2]!, 10);
    if (major < PYTHON_MIN_MAJOR) continue;
    if (major === PYTHON_MIN_MAJOR && minor < PYTHON_MIN_MINOR) continue;
    return { found: true, command, argv, version: `${major}.${minor}` };
  }
  return {
    found: false,
    command: "python",
    argv: [],
    message: `Python ${PYTHON_MIN_MAJOR}.${PYTHON_MIN_MINOR}+ not found on PATH`,
  };
}

function checkUv(): ToolCheck {
  const res = spawnSync("uv", ["--version"], { encoding: "utf-8" });
  if (res.status !== 0) {
    return { found: false, command: "uv", argv: [], message: "uv not found on PATH" };
  }
  return { found: true, command: "uv", argv: [], version: (res.stdout || "").trim() };
}

function managedSidecarDir(): string {
  return path.join(os.homedir(), ".docchat", "sidecar");
}

export function managedVenvPython(target: string): string {
  return process.platform === "win32"
    ? path.join(target, ".venv", "Scripts", "python.exe")
    : path.join(target, ".venv", "bin", "python");
}

/**
 * Run the setup flow. Surfaces every failure mode as a vscode error
 * notification with an actionable button (Open Install Page / View Logs).
 *
 * The function is idempotent: re-running cleans the target dir and reinstalls,
 * so a corrupted venv can always be repaired by running the command again.
 */
export async function runSetupSidecar(
  context: vscode.ExtensionContext,
  outputChannel: vscode.OutputChannel
): Promise<void> {
  outputChannel.show(true);
  outputChannel.appendLine("[setup] starting sidecar install");

  const python = checkPython();
  if (!python.found) {
    const choice = await vscode.window.showErrorMessage(
      `DocChat needs Python ${PYTHON_MIN_MAJOR}.${PYTHON_MIN_MINOR}+ on PATH. ${python.message}`,
      "Open Install Page",
      "Cancel"
    );
    if (choice === "Open Install Page") {
      void vscode.env.openExternal(vscode.Uri.parse(PYTHON_INSTALL_URL));
    }
    return;
  }
  outputChannel.appendLine(
    `[setup] python: ${python.command} ${python.argv.join(" ")} (${python.version})`
  );

  const uv = checkUv();
  if (!uv.found) {
    const choice = await vscode.window.showErrorMessage(
      "DocChat needs the 'uv' Python package manager on PATH. Install it once, then re-run setup.",
      "Open Install Page",
      "Cancel"
    );
    if (choice === "Open Install Page") {
      void vscode.env.openExternal(vscode.Uri.parse(UV_INSTALL_URL));
    }
    return;
  }
  outputChannel.appendLine(`[setup] uv: ${uv.version}`);

  const bundledSource = path.join(context.extensionPath, "sidecar-source");
  if (!existsSync(bundledSource)) {
    void vscode.window.showErrorMessage(
      `DocChat setup failed: bundled sidecar source not found at ${bundledSource}. ` +
        "This is a packaging bug - please file an issue at " +
        "https://github.com/AshwinUgale/docchat/issues."
    );
    return;
  }

  const target = managedSidecarDir();
  outputChannel.appendLine(`[setup] target install dir: ${target}`);

  try {
    await vscode.window.withProgress(
      {
        location: vscode.ProgressLocation.Notification,
        title: "DocChat: setting up sidecar",
        cancellable: false,
      },
      async (progress) => {
        progress.report({ message: "copying sidecar source" });
        if (existsSync(target)) {
          outputChannel.appendLine("[setup] cleaning previous install");
          rmSync(target, { recursive: true, force: true });
        }
        mkdirSync(target, { recursive: true });
        cpSync(bundledSource, target, { recursive: true });

        progress.report({ message: "running uv sync (this downloads ~200MB)" });
        outputChannel.appendLine("[setup] running: uv sync");
        const syncRes = spawnSync("uv", ["sync"], {
          cwd: target,
          encoding: "utf-8",
        });
        if (syncRes.stdout) outputChannel.append(syncRes.stdout);
        if (syncRes.stderr) outputChannel.append(syncRes.stderr);
        if (syncRes.status !== 0) {
          throw new Error(`uv sync failed with exit code ${syncRes.status}`);
        }

        progress.report({ message: "verifying install" });
        const venvPy = managedVenvPython(target);
        if (!existsSync(venvPy)) {
          throw new Error(`venv python not found at ${venvPy} after uv sync`);
        }
        const checkRes = spawnSync(
          venvPy,
          [
            "-c",
            "import docchat_sidecar; print(docchat_sidecar.__version__)",
          ],
          { encoding: "utf-8" }
        );
        if (checkRes.status !== 0) {
          if (checkRes.stderr) outputChannel.append(checkRes.stderr);
          throw new Error("docchat_sidecar import check failed after install");
        }
        const installedVersion = (checkRes.stdout || "").trim();
        outputChannel.appendLine(
          `[setup] verified docchat_sidecar ${installedVersion}`
        );

        progress.report({ message: "writing settings" });
        // Global scope so the path persists across workspaces. The user can
        // override per-workspace if they want to point at a different venv.
        await vscode.workspace
          .getConfiguration("docchat")
          .update("sidecarPython", venvPy, vscode.ConfigurationTarget.Global);
        outputChannel.appendLine(
          `[setup] docchat.sidecarPython -> ${venvPy}`
        );

        outputChannel.appendLine("[setup] done");
      }
    );
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    outputChannel.appendLine(`[setup] failed: ${message}`);
    void vscode.window.showErrorMessage(
      `DocChat setup failed: ${message}. See the DocChat output channel for details.`
    );
    return;
  }

  const choice = await vscode.window.showInformationMessage(
    "DocChat sidecar installed. You can now open the chat panel.",
    "Open Chat Panel"
  );
  if (choice === "Open Chat Panel") {
    void vscode.commands.executeCommand("docchat.openPanel");
  }
}
