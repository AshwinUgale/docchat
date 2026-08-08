# Security policy

## Reporting a vulnerability

Please open a private security advisory on the GitHub repository, or contact the
maintainer directly, rather than filing a public issue. We aim to acknowledge
within a few days.

## Threat model & trust boundaries

DocChat is a VS Code extension plus a local Python sidecar. Everything runs on your
machine.

- **The sidecar is local-only.** The extension spawns the Python sidecar on activation and
  kills it on deactivation; it listens on `localhost` on a random port and is not exposed to
  the network.
- **Secrets live in a local, gitignored `.env`.** API keys are read via `python-dotenv` and
  are never committed. Do not paste keys into code, issues, or logs.
- **It reads your code and fetches library docs.** Indexing reads files in your workspace;
  the indexer fetches public documentation for your pinned library versions over the network.
  Treat indexed content and fetched docs as data.
- **The vector store (Qdrant) runs locally** via `docker-compose` and holds your embedded
  content on your machine.

If you find a way to make the sidecar bind beyond localhost, leak a key, or execute
untrusted content from a fetched doc or indexed file, please report it.
