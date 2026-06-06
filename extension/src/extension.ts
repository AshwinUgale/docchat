/**
 * DocChat VS Code extension entry point.
 *
 * v1.0.1 - adds docchat.setupSidecar command and a recovery path when the
 * sidecar exits before printing its port line (most common cause on a fresh
 * Marketplace install: the docchat_sidecar Python module isn't installed yet).
 */

import * as fs from "node:fs";
import * as path from "node:path";
import * as vscode from "vscode";

import { type SidecarHandle, killSidecar, spawnSidecar } from "./sidecar";
import { runSetupSidecar } from "./setup";

let outputChannel: vscode.OutputChannel | undefined;
let activeSidecar: SidecarHandle | undefined;
let activePanel: vscode.WebviewPanel | undefined;

const SIDECAR_NOT_INSTALLED_RE = /sidecar exited before port line/i;

export function activate(context: vscode.ExtensionContext): void {
  outputChannel = vscode.window.createOutputChannel("DocChat");
  context.subscriptions.push(outputChannel);

  const openDisposable = vscode.commands.registerCommand(
    "docchat.openPanel",
    () => openChatPanel(context).catch((err: unknown) => {
      const message = err instanceof Error ? err.message : String(err);
      outputChannel?.appendLine(`[docchat] error: ${message}`);
      if (SIDECAR_NOT_INSTALLED_RE.test(message)) {
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
  if (activePanel) {
    activePanel.reveal();
    return;
  }
  if (!outputChannel) {
    throw new Error("output channel not initialised");
  }

  activeSidecar = await spawnSidecar(context.extensionPath, outputChannel);
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
