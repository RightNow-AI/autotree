## Playground quickstart

1. From `serve/`, install the local package and its runtime requirements with `uv sync --extra dev`.
2. Start the honest demo with `uv run autotree serve --engine treekv --model gpt2` (add `--device cuda --dtype bfloat16` on a GPU box).
3. Open `/playground`, run a prompt, and watch real branches grow, get pruned, and resolve live. (Merge events render when an engine emits them; the TreeKV engine does not emit merges yet.)

## Graceful shutdown

On cooperative `SIGTERM` or Ctrl+C shutdown, the server stops admitting new
generations and waits for already-admitted streams to finish before the app
lifespan exits. This gives connected clients time to receive their terminal
`done`/`error` event and `[DONE]` sentinel.

Windows cannot guarantee this for forced process termination (`taskkill /F`), a
closed console host, interpreter crashes, or client disconnects. Those cases can
cut a stream before its terminal frames; callers must still use request timeouts
and treat a missing terminal event as an incomplete result.
