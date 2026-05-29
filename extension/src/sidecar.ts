/**
 * Sidecar subprocess manager.
 *
 * Responsibilities:
 *   1. Resolve which Python interpreter to use (settings override > sidecar/.venv > PATH).
 *   2. Spawn `python -m docchat_sidecar --port 0` so the OS picks a free port.
 *   3. Parse the first stdout line `DOCCHAT_SIDECAR_PORT=<N>` to get the bound port.
 *   4. Poll `/health` until the FastAPI app is actually accepting connections.
 *   5. Return {process, port} so the caller can open the chat panel.
 *
 * Spawn + parse + health-poll is the multi-process plumbing this milestone exists
 * to prove. The webview never talks to subprocess stdout — it goes through
 * WebSocket only — so this module is the single point of trust for the lifecycle.
 *
 * Why we read stdout for the port instead of a fixed port: VS Code can have N
 * windows open at once, each with its own sidecar; fixed ports collide. The OS
 * pick + stdout handshake is the standard pattern Continue.dev, Cody, Tabby use.
 */

import { spawn, type ChildProcess } from "node:child_process";
import * as fs from "node:fs";
import * as http from "node:http";
import * as path from "node:path";
import * as vscode from "vscode";

export interface SidecarHandle {
  process: ChildProcess;
  port: number;
}

const PORT_LINE_RE = /^DOCCHAT_SIDECAR_PORT=(\d+)\s*$/;
const HEALTH_TIMEOUT_MS = 5_000;
const HEALTH_POLL_INTERVAL_MS = 100;

/**
 * Resolve which Python to spawn.
 *
 * Priority:
 *   1. `docchat.sidecarPython` setting if explicitly set to something other than "python".
 *   2. `<repo>/sidecar/.venv/Scripts/python.exe` (Windows) or `.../bin/python` (Unix)
 *      when running F5-style from a dev workspace that contains the sidecar source.
 *   3. `python` on PATH (works if `pip install docchat-sidecar` is on the system).
 */
function resolvePython(extensionPath: string): string {
  const setting = vscode.workspace
    .getConfiguration("docchat")
    .get<string>("sidecarPython", "python");
  if (setting && setting !== "python") {
    return setting;
  }
  // Walk up from the extension folder looking for ../sidecar/.venv.
  const candidateRoots = [
    path.join(extensionPath, "..", "sidecar"),
    path.join(extensionPath, "sidecar"),
  ];
  for (const root of candidateRoots) {
    const venvPython = process.platform === "win32"
      ? path.join(root, ".venv", "Scripts", "python.exe")
      : path.join(root, ".venv", "bin", "python");
    if (fs.existsSync(venvPython)) {
      return venvPython;
    }
  }
  return "python";
}

function getHealth(port: number): Promise<boolean> {
  return new Promise((resolve) => {
    const req = http.get(
      { host: "127.0.0.1", port, path: "/health", timeout: 500 },
      (res) => {
        res.resume();
        resolve(res.statusCode === 200);
      }
    );
    req.on("error", () => resolve(false));
    req.on("timeout", () => {
      req.destroy();
      resolve(false);
    });
  });
}

async function waitForHealth(port: number): Promise<void> {
  const deadline = Date.now() + HEALTH_TIMEOUT_MS;
  while (Date.now() < deadline) {
    if (await getHealth(port)) {
      return;
    }
    await new Promise((r) => setTimeout(r, HEALTH_POLL_INTERVAL_MS));
  }
  throw new Error(`sidecar /health did not respond within ${HEALTH_TIMEOUT_MS}ms`);
}

/**
 * Spawn the Python sidecar. Resolves once `/health` returns 200.
 *
 * Caller is responsible for calling `killSidecar(handle)` to clean up.
 */
export async function spawnSidecar(
  extensionPath: string,
  outputChannel: vscode.OutputChannel
): Promise<SidecarHandle> {
  const python = resolvePython(extensionPath);
  outputChannel.appendLine(`[docchat] spawning sidecar: ${python} -m docchat_sidecar --port 0`);

  const proc = spawn(python, ["-m", "docchat_sidecar", "--port", "0"], {
    stdio: ["ignore", "pipe", "pipe"],
    // Detached so we can kill the whole process group; required because uvicorn
    // spawns worker threads we'd otherwise orphan on Ctrl+C.
    detached: false,
  });

  // Forward stderr to the output channel for debugging.
  proc.stderr?.on("data", (chunk: Buffer) => {
    outputChannel.append(`[sidecar stderr] ${chunk.toString()}`);
  });

  const port = await readPortFromStdout(proc, outputChannel);

  // Continue forwarding stdout AFTER the port line so subsequent logs land in
  // the output channel. The reader in readPortFromStdout drains the first line
  // and detaches; here we re-attach for the lifetime of the process.
  proc.stdout?.on("data", (chunk: Buffer) => {
    outputChannel.append(`[sidecar stdout] ${chunk.toString()}`);
  });

  await waitForHealth(port);
  outputChannel.appendLine(`[docchat] sidecar ready on port ${port}`);

  return { process: proc, port };
}

/**
 * Read stdout line-by-line until we see DOCCHAT_SIDECAR_PORT=<N>.
 *
 * We can't just `await proc.stdout.read()` — uvicorn's startup banner can
 * arrive interleaved with our port line if logging is configured aggressively,
 * so we line-buffer until the regex matches.
 */
function readPortFromStdout(
  proc: ChildProcess,
  outputChannel: vscode.OutputChannel
): Promise<number> {
  return new Promise((resolve, reject) => {
    if (!proc.stdout) {
      reject(new Error("sidecar process has no stdout"));
      return;
    }
    let buffer = "";
    const onData = (chunk: Buffer): void => {
      buffer += chunk.toString();
      const lines = buffer.split(/\r?\n/);
      buffer = lines.pop() ?? "";
      for (const line of lines) {
        const match = PORT_LINE_RE.exec(line);
        if (match) {
          proc.stdout?.off("data", onData);
          resolve(Number.parseInt(match[1]!, 10));
          return;
        }
        outputChannel.append(`[sidecar stdout] ${line}\n`);
      }
    };
    proc.stdout.on("data", onData);
    proc.once("exit", (code) =>
      reject(new Error(`sidecar exited before port line (code ${code})`))
    );
  });
}

/**
 * Kill the sidecar process. Called from extension.deactivate() and panel dispose.
 */
export function killSidecar(handle: SidecarHandle): void {
  if (handle.process.exitCode === null && handle.process.signalCode === null) {
    handle.process.kill("SIGTERM");
  }
}
