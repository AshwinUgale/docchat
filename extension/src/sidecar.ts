/**
 * Sidecar subprocess manager.
 *
 * Priority for resolving the Python interpreter (v1.0.1):
 *   1. docchat.sidecarPython setting (other than literal "python").
 *   2. ~/.docchat/sidecar/.venv/... - managed venv from `DocChat: Set up sidecar`.
 *   3. <repo>/sidecar/.venv/... - dev workflow (F5 from source).
 *   4. python on PATH - last-ditch fallback.
 */

import { spawn, type ChildProcess } from "node:child_process";
import * as fs from "node:fs";
import * as http from "node:http";
import * as os from "node:os";
import * as path from "node:path";
import * as vscode from "vscode";

export interface SidecarHandle {
  process: ChildProcess;
  port: number;
}

const PORT_LINE_RE = /^DOCCHAT_SIDECAR_PORT=(\d+)\s*$/;
const HEALTH_TIMEOUT_MS = 5_000;
const HEALTH_POLL_INTERVAL_MS = 100;

function resolvePython(extensionPath: string): string {
  const setting = vscode.workspace
    .getConfiguration("docchat")
    .get<string>("sidecarPython", "python");
  if (setting && setting !== "python") {
    return setting;
  }
  const candidateRoots = [
    path.join(os.homedir(), ".docchat", "sidecar"),
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

export async function spawnSidecar(
  extensionPath: string,
  outputChannel: vscode.OutputChannel
): Promise<SidecarHandle> {
  const python = resolvePython(extensionPath);
  outputChannel.appendLine(`[docchat] spawning sidecar: ${python} -m docchat_sidecar --port 0`);

  const proc = spawn(python, ["-m", "docchat_sidecar", "--port", "0"], {
    stdio: ["ignore", "pipe", "pipe"],
    detached: false,
  });

  proc.stderr?.on("data", (chunk: Buffer) => {
    outputChannel.append(`[sidecar stderr] ${chunk.toString()}`);
  });

  const port = await readPortFromStdout(proc, outputChannel);

  proc.stdout?.on("data", (chunk: Buffer) => {
    outputChannel.append(`[sidecar stdout] ${chunk.toString()}`);
  });

  await waitForHealth(port);
  outputChannel.appendLine(`[docchat] sidecar ready on port ${port}`);

  return { process: proc, port };
}

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

export function killSidecar(handle: SidecarHandle): void {
  if (handle.process.exitCode === null && handle.process.signalCode === null) {
    handle.process.kill("SIGTERM");
  }
}
