## Playground quickstart

1. From `serve/`, install the local package and its runtime requirements with `uv sync --extra dev`.
2. Start the honest CPU demo with `uv run autotree serve --engine treekv --model gpt2`.
3. Open `/playground`, run a prompt, and watch real branches grow, get pruned, and resolve live. (Merge events render when an engine emits them; the TreeKV engine does not emit merges yet.)
