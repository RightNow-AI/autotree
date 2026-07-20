# Tree-KV capacity behavior

For `autotree serve --engine treekv`, the default page limit is:

`ceil(model_context_tokens / page_size * kv_branch_headroom)`

The default `kv_branch_headroom` is `1.5`. `--kv-pages` overrides the derived
limit, and `--kv-branch-headroom` changes the multiplier.

Prompt admission and decode exhaustion are exposed as
`kv_capacity_exhausted` errors instead of server errors. A non-streaming
request receives HTTP 429. Once an SSE response has started, the server emits
an `error` event and closes the stream normally.

The engine does not prune branches on its own when allocation fails. The Rust
scheduler owns branch liveness and emits `Kill` commands; injecting an
engine-side kill would leave scheduler and KV state inconsistent. Capacity is
therefore recovered only through scheduler-issued pruning, otherwise the
request ends with the typed error above.
