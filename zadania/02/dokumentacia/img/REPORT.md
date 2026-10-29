# Cognee local hosting and repository use

**Research date:** 23 September 2026  
**Scope:** How to download and run [Cognee](https://www.cognee.ai/) locally, use it with Codex, and index other code repositories.

## Recommendation

Start with the [Cognee Codex plugin](https://github.com/topoteretes/cognee-integrations/tree/main/integrations/codex) if Codex is your main interface. It installs and starts a local Cognee API, captures agent memory, and indexes Git repositories opened in Codex. Each repository gets its own code-graph dataset. Use a [self-hosted API and MCP deployment](https://github.com/topoteretes/cognee#deploy-cognee) when several applications or AI clients need a continuously running shared service.

Cognee is open source and can keep its graph, vector, and metadata stores on your machine. Local hosting alone does **not** make Cognee's model inference local: its default LLM and embedding provider is OpenAI. Configure both providers for Ollama to keep Cognee's model calls on your machine. Your coding agent's own model connection is a separate setting. [Cognee README](https://github.com/topoteretes/cognee#quickstart) · [Local model setup](https://docs.cognee.ai/guides/local-setup)

## What is installed on this Mac

Read-only command checks found `git`, `codex`, `python3`, `uv`, and `docker`. `ollama` was unavailable. The installed `codex` command exposes plugin management. No installation or configuration was performed during this research.

## Option A: Codex plugin with a local Cognee service

The [official plugin guide](https://github.com/topoteretes/cognee-integrations/tree/main/integrations/codex) gives these commands:

```sh
codex features enable hooks
codex plugin marketplace add topoteretes/cognee-integrations --ref main
codex plugin add cognee@cognee
```

The plugin's default local mode bootstraps a Cognee API at `http://localhost:8011`. It reads configuration from `~/.cognee/.env` and starts its own Python runtime. Start a new Codex session after configuration. Its hooks capture prompts, tool traces, and answers, then recall and sync relevant session memory. [Plugin setup and behavior](https://github.com/topoteretes/cognee-integrations/tree/main/integrations/codex)

### Use Ollama for fully local model calls

Install Ollama, run its service, and download the models:

```sh
ollama pull llama3.1:8b
ollama pull nomic-embed-text:latest
```

Put the following values in `~/.cognee/.env` before starting a new Codex session:

```dotenv
LLM_PROVIDER="ollama"
LLM_MODEL="llama3.1:8b"
LLM_ENDPOINT="http://localhost:11434"
LLM_API_KEY="ollama"

EMBEDDING_PROVIDER="ollama"
EMBEDDING_MODEL="nomic-embed-text:latest"
EMBEDDING_ENDPOINT="http://localhost:11434/api/embed"
EMBEDDING_DIMENSIONS="768"
HUGGINGFACE_TOKENIZER="nomic-ai/nomic-embed-text-v1.5"
```

`LLM_API_KEY="ollama"` is a placeholder required by Cognee; Ollama does not validate it. Set **both** provider blocks so embeddings do not fall back to OpenAI. The model guide also documents Ollama with Fastembed as an alternative. [Local setup guide](https://docs.cognee.ai/guides/local-setup)

### Index repositories

Open Codex inside a Git repository. The plugin indexes it in the background at session start and refreshes the graph after a turn changes the working tree. For an additional local path or Git URL, its explicit index command is:

```sh
${CODEX_PLUGIN_ROOT}/scripts/cognee-index-repo.sh <repo-path-or-git-url> [--dataset <name>] [--index-vectors] [--wait <seconds>]
```

The plugin creates a dataset named like `codebase-<repo-name>-<digest>` for each repository. Its default local mode indexes the working tree, including uncommitted changes. A remote server indexes a cloned Git URL and therefore sees pushed commits. The indexer writes a `.enola/` snapshot **inside the indexed repository**; add `.enola/` to that repository's `.gitignore` or global Git excludes if needed. Automatic indexing skips non-Git directories and repositories with more than 3,000 source files; explicit indexing has no size cap. [Code graph guide](https://github.com/topoteretes/cognee-integrations/tree/main/integrations/codex#code-graph)

The code graph records symbols, calls, imports, endpoints, and dependencies. Building this deterministic graph makes no LLM or embedding calls. Optional vector indexing and other memory operations can use the configured models. [Code graph guide](https://github.com/topoteretes/cognee-integrations/tree/main/integrations/codex#code-graph)

### Keep agent memory scoped by project

The Codex and Claude Code plugins default to a shared `agent_sessions` dataset. Set `COGNEE_PLUGIN_DATASET` before launching Codex to use a project-specific memory dataset:

```sh
COGNEE_PLUGIN_DATASET="my-project-memory" codex
```

The plugin recalls from its active memory dataset. Its repository code graphs have their own datasets. The plugin also supports switching the memory dataset during a session. [Plugin dataset guide](https://github.com/topoteretes/cognee-integrations/tree/main/integrations/codex#dataset)

## Option B: Host Cognee as a shared local service

For a service used by multiple apps, clone the [Cognee repository](https://github.com/topoteretes/cognee), copy its environment template, configure model providers and storage, then start the API, UI, and MCP profiles:

```sh
git clone https://github.com/topoteretes/cognee.git
cd cognee
cp .env.template .env
# Configure .env before starting the services.
docker compose --profile ui --profile mcp up
```

The documented default ports are API **8000**, UI **3000**, and MCP **8001**. For an API-only trial using a prebuilt image, Cognee provides a [minimal Docker Compose guide](https://github.com/topoteretes/cognee/blob/main/docs/minimal-docker-compose.md), including a named-volume example for persistence. Without a persistent volume in that minimal example, removing the container removes the stored data. Review authentication before exposing the service beyond your own machine. [Deployment guide](https://github.com/topoteretes/cognee#deploy-cognee)

If Cognee runs **inside Docker** while Ollama runs on the Mac, use `http://host.docker.internal:11434` for the LLM endpoint and `http://host.docker.internal:11434/api/embed` for the embedding endpoint: `localhost` inside the container refers to the container. [Ollama troubleshooting](https://docs.cognee.ai/guides/local-setup#troubleshooting)

Python applications can use the [Cognee Python API](https://docs.cognee.ai/), other applications can use the [REST API](https://docs.cognee.ai/), and compatible AI clients can connect through [Cognee MCP](https://docs.cognee.ai/cognee-mcp/mcp-overview). In MCP's API mode, multiple clients connect to one Cognee backend and share its graph. [MCP architecture](https://docs.cognee.ai/cognee-mcp/mcp-overview)

## Direct Python use for another repository

Cognee supports explicit code-graph ingestion from a local repository path or Git URL. Its API requires `content_type="code"` for this route; it does not infer it from every `remember()` input. Use a distinct dataset for each repository. [Cognee `remember()` implementation](https://github.com/topoteretes/cognee/blob/main/cognee/api/v1/remember/remember.py) · [Official example](https://github.com/topoteretes/cognee/blob/main/examples/demos/company_brain/company_brain_demo.py)

```python
import asyncio
import cognee

async def main():
    result = await cognee.remember(
        "/absolute/path/to/your-repo",
        dataset_name="codebase-your-repo",
        content_type="code",
    )
    print(result)

asyncio.run(main())
```

The [Cognee README](https://github.com/topoteretes/cognee#quickstart) lists Python 3.10–3.14 for the core package. This Python example runs Cognee in process; use the shared API or MCP service when several clients must access the same local database concurrently. The MCP docs distinguish standalone instances, which keep separate data, from API mode, which shares one backend. [MCP modes](https://docs.cognee.ai/cognee-mcp/mcp-overview)

## Suggested path

1. Install Ollama and its two models if fully local model calls are required.
2. Install the Codex plugin and configure `~/.cognee/.env`.
3. Start Codex in one small Git repository and confirm its code graph appears.
4. Index other repositories explicitly or open Codex in each one.
5. Move to a persistent Docker API/MCP deployment if independent applications need one continuously running shared memory service.

**Sources:** [Cognee main repository](https://github.com/topoteretes/cognee) · [Codex plugin](https://github.com/topoteretes/cognee-integrations/tree/main/integrations/codex) · [Local model configuration](https://docs.cognee.ai/guides/local-setup) · [Minimal Docker Compose](https://github.com/topoteretes/cognee/blob/main/docs/minimal-docker-compose.md) · [MCP overview](https://docs.cognee.ai/cognee-mcp/mcp-overview)
