/**
 * DocChat VS Code extension entry point.
 *
 * v1.0.1 - adds docchat.setupSidecar command and a recovery path when the
 * sidecar exits before printing its port line (most common cause on a fresh
 * Marketplace install: the docchat_sidecar Python module isn't installed yet).
 *
 * v1.0.2 - adds docchat.setOpenAIKey command using vscode.SecretStorage.
 * The extension reads the stored key on every panel-open and passes it to
 * the spawned sidecar via env override, so users never need to put the key
 * in their shell rc-file. Detects the "no key" sidecar exit signal and
 * surfaces a one-click "Set Key" recovery action.
 */

import * as fs from "node:fs";
import * as path from "node:path";
import * as vscode from "vscode";

import { type SidecarHandle, killSidecar, spawnSidecar } from "./sidecar";
import { runSetupSidecar } from "./setup";

const OPENAI_API_KEY_SECRET = "docchat.openaiApiKey";

let outputChannel: vscode.OutputChannel | undefined;
let activeSidecar: SidecarHandle | undefined;
let activePanel: vscode.WebviewPanel | undefined;

const SIDECAR_NOT_INSTALLED_RE = /sidecar exited before port line/i;
const KEY_NOT_SET_RE = /OPENAI_API_KEY is not set/i;

export function activate(context: vscode.ExtensionContext): void {
  outputChannel = vscode.window.createOutputChannel("DocChat");
  context.subscriptions.push(outputChannel);

  const openDisposable = vscode.commands.registerCommand(
    "docchat.openPanel",
    () => openChatPanel(context).catch((err: unknown) => {
      const message = err instanceof Error ? err.message : String(err);
      outputChannel?.appendLine(`[docchat] error: ${message}`);
      // v1.0.2 - key-missing recovery path takes precedence over the
      // sidecar-not-installed one (a fresh install can also hit the
      // key-missing path right after setup completes).
      if (KEY_NOT_SET_RE.test(message)) {
        void vscode.window
          .showErrorMessage(
            "DocChat needs an OpenAI API key. It powers query + index-time embeddings.",
            "Set Key",
            "View Logs"
          )
          .then((choice) => {
            if (choice === "Set Key") {
              void vscode.commands.executeCommand("docchat.setOpenAIKey");
            } else if (choice === "View Logs") {
              outputChannel?.show(true);
            }
          });
      } else if (SIDECAR_NOT_INSTALLED_RE.test(message)) {
        void vscode.window
          .showErrorMessage(
            `DocChat: ${message}. The sidecar may not be installed yet.`,
            "Run Setup",
            "View Logs"
          )
          .then((choice) => {
            if (choice === "Run Setup") {
              void vscode.commands.executeCommand("docchat.setupSidecar");
            } else if (choice === "View Logs") {
              outputChannel?.show(true);
            }
          });
      } else {
        void vscode.window.showErrorMessage(`DocChat: ${message}`);
      }
    })
  );
  context.subscriptions.push(openDisposable);

  // v1.0.1 - per-user install of the bundled sidecar source into
  // ~/.docchat/sidecar/. See setup.ts for the full flow.
  const setupDisposable = vscode.commands.registerCommand(
    "docchat.setupSidecar",
    () => {
      if (!outputChannel) return;
      void runSetupSidecar(context, outputChannel).catch((err: unknown) => {
        const message = err instanceof Error ? err.message : String(err);
        outputChannel?.appendLine(`[docchat] setup error: ${message}`);
        void vscode.window.showErrorMessage(`DocChat setup: ${message}`);
      });
    }
  );
  context.subscriptions.push(setupDisposable);

  // v1.0.2 - prompt for the key with masked input, store in SecretStorage.
  // SecretStorage is per-user, encrypted by VS Code's keychain integration,
  // and never written to settings.json. Re-running the command overwrites.
  const setKeyDisposable = vscode.commands.registerCommand(
    "docchat.setOpenAIKey",
    async () => {
      const existing = await context.secrets.get(OPENAI_API_KEY_SECRET);
      const key = await vscode.window.showInputBox({
        title: "DocChat: OpenAI API Key",
        prompt: existing
          ? "Replace the currently stored key. Leave empty to keep existing."
          : "Used for query + index-time embeddings. Stored in VS Code SecretStorage (encrypted, per-user).",
        placeHolder: "sk-...",
        password: true,
        ignoreFocusOut: true,
        validateInput: (v) => {
          if (!v) return null; // allow empty to abort
          if (!v.startsWith("sk-")) return "OpenAI keys start with 'sk-'.";
          if (v.length < 20) return "Looks too short to be a real OpenAI key.";
          return null;
        },
      });
      if (!key) {
        outputChannel?.appendLine("[docchat] setOpenAIKey cancelled");
        return;
      }
      await context.secrets.store(OPENAI_API_KEY_SECRET, key);
      outputChannel?.appendLine("[docchat] OpenAI key stored in SecretStorage");
      const choice = await vscode.window.showInformationMessage(
        "OpenAI key saved. Open the DocChat panel?",
        "Open Chat Panel"
      );
      if (choice === "Open Chat Panel") {
        void vscode.commands.executeCommand("docchat.openPanel");
      }
    }
  );
  context.subscriptions.push(setKeyDisposable);
}

export function deactivate(): void {
  if (activeSidecar) {
    killSidecar(activeSidecar);
    activeSidecar = undefined;
  }
  activePanel?.dispose();
  activePanel = undefined;
}

async function openChatPanel(context: vscode.ExtensionContext): Promise<void> {
  // Reuse the existing panel if it's still open.
  if (activePanel) {
    activePanel.reveal();
    return;
  }
  if (!outputChannel) {
    throw new Error("output channel not initialised");
  }

  // v1.0.2 - pull the stored OpenAI key (if any) and pass it to the sidecar
  // process. Falls through to whatever the user's shell has set if the
  // SecretStorage entry is empty; the sidecar's fail-fast check will catch
  // the case where neither source has it and trigger the Set Key flow.
  const storedKey = await context.secrets.get(OPENAI_API_KEY_SECRET);
  const envOverrides: Record<string, string | undefined> = {};
  if (storedKey) {
    envOverrides.OPENAI_API_KEY = storedKey;
  }

  activeSidecar = await spawnSidecar(context.extensionPath, outputChannel, envOverrides);
  const port = activeSidecar.port;

  const panel = vscode.window.createWebviewPanel(
    "docchat.chat",
    "DocChat",
    vscode.ViewColumn.Beside,
    {
      enableScripts: true,
      retainContextWhenHidden: true,
      localResourceRoots: [
        vscode.Uri.file(path.join(context.extensionPath, "webview")),
      ],
    }
  );
  activePanel = panel;

  panel.webview.html = loadWebviewHtml(context.extensionPath);

  void panel.webview.postMessage({ type: "sidecarPort", port });

  panel.webview.onDidReceiveMessage((msg: unknown) => {
    if (!msg || typeof msg !== "object") return;
    const m = msg as Record<string, unknown>;
    if (m.type === "openCitation") {
      const url = typeof m.source_url === "string" ? m.source_url : null;
      if (!url) {
        outputChannel?.appendLine("[docchat] openCitation skipped: no source_url");
        return;
      }
      if (!url.startsWith("http://") && !url.startsWith("https://")) {
        outputChannel?.appendLine(`[docchat] openCitation rejected non-http URL: ${url}`);
        return;
      }
      void vscode.env.openExternal(vscode.Uri.parse(url));
    }
  });

  panel.onDidDispose(() => {
    activePanel = undefined;
    if (activeSidecar) {
      killSidecar(activeSidecar);
      activeSidecar = undefined;
    }
  });
}

function loadWebviewHtml(extensionPath: string): string {
  const htmlPath = path.join(extensionPath, "webview", "index.html");
  return fs.readFileSync(htmlPath, "utf-8");
}
