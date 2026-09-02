# Working on this repository

Rules for anyone — or any coding agent — changing this code.

## Before you change anything
- Run `uv sync --extra dev` and `uv run pytest -q`. Everything must be green before you start.
- Read `pyproject.toml`. Every dependency is pinned to an exact version on purpose.

## Dependencies
- Exact versions only. No `>=`, `~=` or `^`. The pipeline rejects them.
- Adding a dependency is a decision, not a side effect: say why in the pull request.
- After any change to dependencies run `uv lock` and commit `uv.lock`.

## Secrets and data
- Never commit `.env`, keys, tokens, or sample documents containing personal data.
- `.env.example` holds placeholders only, never a real credential. Keep it that way.

## Tests
- A change to behaviour comes with a test that fails without the change.
- Tests must not call a model provider or a vector database. Fake the client.

## Pull requests
- Every change is opened as a pull request and read by a person before it is merged.
- State whether an agent wrote the change and what you personally verified.
- Read the diff in blast-radius order: dependencies, config, then code, then tests.
