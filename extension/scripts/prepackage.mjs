/**
 * Bundle the Python sidecar source into the .vsix.
 *
 * Runs as part of `vscode:prepublish`, so both `vsce package` and `vsce
 * publish` pick it up automatically. The output folder, ``sidecar-source/``,
 * is a build artifact - .gitignored and regenerated each package.
 *
 * What we copy (the minimal set ``uv sync`` needs):
 *   - sidecar/pyproject.toml  (rewritten: readme path made local)
 *   - sidecar/uv.lock         (pinned dep resolution)
 *   - sidecar/src/            (the docchat_sidecar package)
 *
 * What we generate:
 *   - sidecar-source/README.md  (1-line stub so hatchling has something to read
 *                                when ``uv sync`` builds the editable install)
 *
 * What we deliberately skip:
 *   - sidecar/tests/         (not needed at runtime; bloats the .vsix)
 *   - sidecar/.venv/         (per-user; setup.ts recreates inside the venv)
 *   - sidecar/.mypy_cache/   (dev artifact)
 *   - sidecar/.pytest_cache/ (dev artifact)
 *   - sidecar/.ruff_cache/   (dev artifact)
 *   - __pycache__/           (bytecode; regenerated on first import)
 */

import { cpSync, existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const repoRoot = join(__dirname, "..", "..");
const sidecarSrc = join(repoRoot, "sidecar");
const bundleDst = join(__dirname, "..", "sidecar-source");

if (!existsSync(sidecarSrc)) {
  console.error(`[prepackage] sidecar source not found at ${sidecarSrc}`);
  process.exit(1);
}

// Clean previous bundle so deletions upstream propagate.
if (existsSync(bundleDst)) {
  rmSync(bundleDst, { recursive: true, force: true });
}
mkdirSync(bundleDst, { recursive: true });

// 1. pyproject.toml - rewrite the readme path so the bundled standalone
// install doesn't try to read ../README.md (which won't exist at
// ~/.docchat/sidecar/).
const pyprojectSrc = readFileSync(join(sidecarSrc, "pyproject.toml"), "utf-8");
const pyprojectRewritten = pyprojectSrc.replace(
  /readme\s*=\s*["']\.\.\/README\.md["']/,
  'readme = "README.md"'
);
writeFileSync(join(bundleDst, "pyproject.toml"), pyprojectRewritten, "utf-8");

// 2. uv.lock - keep the pinned resolution for reproducible installs.
const uvLockPath = join(sidecarSrc, "uv.lock");
if (existsSync(uvLockPath)) {
  cpSync(uvLockPath, join(bundleDst, "uv.lock"));
} else {
  console.warn("[prepackage] uv.lock missing; install will resolve fresh");
}

// 3. src/ - the package itself. Filter out bytecode and editor caches.
const srcFilter = (src) => {
  const segs = src.split(/[\\/]/);
  if (segs.includes("__pycache__")) return false;
  if (segs.some((s) => s.endsWith(".pyc"))) return false;
  return true;
};
cpSync(join(sidecarSrc, "src"), join(bundleDst, "src"), {
  recursive: true,
  filter: srcFilter,
});

// 4. Stub README so hatchling's build pass during `uv sync` has something
// to read for the project metadata.
const stubReadme =
  "# docchat-sidecar (bundled)\n\n" +
  "This is the bundled sidecar shipped inside the DocChat VS Code extension. " +
  "It is installed into `~/.docchat/sidecar/` by the `DocChat: Set up sidecar` " +
  "command. The canonical source lives at " +
  "https://github.com/AshwinUgale/docchat.\n";
writeFileSync(join(bundleDst, "README.md"), stubReadme, "utf-8");

console.log(`[prepackage] bundled sidecar -> ${bundleDst}`);
