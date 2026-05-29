/**
 * DocChat VS Code extension entry point.
 *
 * Lifecycle:
 *   activate()    -> registers the docchat.openPanel command
 *   command       -> spawns the sidecar (if not running) and opens the webview
 *   deactivate()  -> kills the sidecar
 *
 * v0.0 ships the stub: command registration only. v0.1 wires up the sidecar
 * subprocess spawn + WebSocket round-trip + webview panel.
 */

import * as vscode from "vscode";

export function activate(context: vscode.ExtensionContext): void {
  const disposable = vscode.commands.registerCommand(
    "docchat.openPanel",
    () => {
      vscode.window.showInformationMessage(
        "DocChat v0.0 — panel + sidecar wiring lands at v0.1."
      );
    }
  );
  context.subscriptions.push(disposable);
}

export function deactivate(): void {
  // v0.1: kill the sidecar subprocess here.
}
