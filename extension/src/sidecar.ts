/**
 * Sidecar subprocess manager.
 *
 * v0.0: stub. The full lifecycle (spawn, wait-for-port, health-check, kill)
 * lands at v0.1 along with the WebSocket round-trip test.
 *
 * Why a separate process: the VS Code extension is TypeScript. The agent
 * loop, vector store, and dogfood libraries (Mneme, ToolPicker) are Python.
 * We bridge over local WebSocket — same pattern Continue.dev, Cody, Tabby
 * use. See .cowork/DECISIONS.md ADR-001 for the full rationale.
 */

import type { ChildProcess } from "node:child_process";

export interface SidecarHandle {
  process: ChildProcess;
  port: number;
}

/**
 * Spawn the Python sidecar. v0.1 implementation will:
 *   1. Resolve the python executable from settings (docchat.sidecarPython)
 *   2. Spawn `python -m docchat_sidecar --port <port>`
 *   3. Stream stdout/stderr into the extension's output channel
 *   4. Poll the WebSocket /health endpoint until ready or timeout
 *   5. Return { process, port } so the caller can open the chat panel
 */
export async function spawnSidecar(): Promise<SidecarHandle> {
  throw new Error("spawnSidecar: not implemented until v0.1");
}

/**
 * Kill the sidecar process. Called from extension.deactivate().
 */
export function killSidecar(handle: SidecarHandle): void {
  handle.process.kill("SIGTERM");
}
