/**
 * DocChat VS Code extension entry point.
 *
 * Lifecycle (v0.1):
 *   activate()           -> register the docchat.openPanel command, create output channel
 *   docchat.openPanel    -> spawn the sidecar (lazy), open a WebviewPanel with the
 *                           chat UI, postMessage the sidecar port into the page
 *   panel.onDidDispose   -> kill the sidecar (one panel = one sidecar at v0.1)
 *   deactivate()         -> kill the sidecar if it's still running
 *
 * The sidecar lifecycle is tied to the panel, not the extension. Closing the
 * panel kills the Python process; reopening the command spawns a fresh one.
 * Multi-panel support (one sidecar shared by N panels) lands later if it's
 * worth the complexity; v0.1 keeps the lifecycle 1:1.
 */

import * as fs from "node:fs";
import * as path from "node:path";
import * as vscode from "vscode";

import { type SidecarHandle, killSidecar, spawnSidecar } from "./sidecar";

let outputChannel: vscode.OutputChannel | undefined;
let activeSidecar: SidecarHandle | undefined;
let activePanel: vscode.WebviewPanel | undefined;

export function activate(context: vscode.ExtensionContext): void {
  outputChannel = vscode.window.createOutputChannel("DocChat");
  context.subscriptions.push(outputChannel);

  const disposable = vscode.commands.registerCommand(
    "docchat.openPanel",
    () => openChatPanel(context).catch((err: unknown) => {
      const message = err instanceof Error ? err.message : String(err);
      outputChannel?.appendLine(`[docchat] error: ${message}`);
      void vscode.window.showErrorMessage(`DocChat: ${message}`);
    })
  );
  context.subscriptions.push(disposable);
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

  // Spawn sidecar first so we have the port before creating the webview.
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

  // Inject the sidecar port. The webview listens for this message before
  // opening its WebSocket — keeping the port out of the URL means no
  // hard-coded values in HTML.
  void panel.webview.postMessage({ type: "sidecarPort", port });

  // v0.7.1 — webview → extension messages. Currently only ``openCitation``:
  // the webview's click-to-open citation chips post a structured message
  // with the source URL, and we open it via ``vscode.env.openExternal``.
  // Settings updates are NOT routed through here — the webview sends
  // them directly to the sidecar over WebSocket since the sidecar is
  // what acts on them.
  panel.webview.onDidReceiveMessage((msg: unknown) => {
    if (!msg || typeof msg !== "object") return;
    const m = msg as Record<string, unknown>;
    if (m.type === "openCitation") {
      const url = typeof m.source_url === "string" ? m.source_url : null;
      if (!url) {
        outputChannel?.appendLine("[docchat] openCitation skipped: no source_url");
        return;
      }
      // Only allow http(s) URLs - we don't want a malicious payload to
      // shell out to file:// or vscode:// via openExternal.
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
