# AutoTree Tree Completions Wire Contract v1

Status: **NORMATIVE** for `autotree-serve`, `autotree-sdk`, and consumers of
`POST /v1/tree/completions`.

Contract version: **1.0.0**. The `/v1` path identifies this major wire version.
Breaking changes require a new major endpoint or an explicitly negotiated wire
version. Additive response fields may be introduced within v1; clients must not
infer semantics from fields that are not specified here.

The key words **MUST**, **MUST NOT**, **SHOULD**, and **MAY** are normative.

## Request

The request body is JSON with these fields:

| Field | Type | Required | Contract |
|---|---|---:|---|
| `model` | string | yes | Non-empty and equal to a model exposed by this serve process. |
| `messages` | array | yes | At least one `{role, content}` object. `role` is a non-empty string and `content` is a string. |
| `tree` | object | yes | Tree-search parameters described below. |
| `stream` | boolean | no | Defaults to `false`. |
| `max_completion_tokens` | integer | no | 1 through 4096. Takes precedence over `max_tokens`. |
| `max_tokens` | integer | no | 1 through 4096. Used only when `max_completion_tokens` is absent. |
| `temperature` | number | no | 0 through 2 inclusive; default 1. |
| `top_p` | number | no | Greater than 0 and at most 1; default 1. |
| `stop` | string or string array | no | Stop sequence or sequences. |
| `n` | integer | no | MUST equal 1; default 1. |
| `seed` | integer or null | no | Sampling seed. |
| `user` | string or null | no | Caller-provided user identifier. |
| `stream_options` | object or null | no | Accepted for OpenAI compatibility. Tree streams always report usage in `done`. |

If neither token-limit field is present, `max_tokens` resolves to 16.

The `tree` object has exactly these fields:

| Field | Type | Required | Contract |
|---|---|---:|---|
| `policy` | string | yes | One of `beam`, `best_first`, or `mcts`. |
| `branches` | integer | yes | 1 through 64. |
| `budget_tokens` | integer | yes | 1 through 1,000,000. |
| `scorer` | string or null | no | Optional scorer identifier. |

The following OpenAI chat fields are explicitly unsupported and MUST produce an
`unsupported_feature` error when present: `audio`, `frequency_penalty`,
`function_call`, `functions`, `logit_bias`, `logprobs`, `modalities`,
`parallel_tool_calls`, `prediction`, `presence_penalty`, `reasoning_effort`,
`response_format`, `tool_choice`, `tools`, `top_logprobs`, and
`web_search_options`. Other unknown top-level request fields are currently
accepted and ignored for compatibility; callers SHOULD NOT rely on them.

Errors use the OpenAI-shaped body:

```json
{
  "error": {
    "message": "human-readable detail",
    "type": "invalid_request_error",
    "param": "field-or-null",
    "code": "machine-readable-code-or-null"
  }
}
```

## Non-stream response

With `stream: false`, the server returns one JSON object:

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 0,
  "model": "served-model-id",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "winning text"},
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 1,
    "completion_tokens": 1,
    "total_tokens": 2
  },
  "tree": {}
}
```

`created` is a Unix timestamp in seconds. `choices` MUST contain exactly one
choice. `finish_reason` is `stop` or `length`. The `tree` object is required and
has the Tree Summary shape defined below. The winning `message.content` MUST be
the concatenation of tokens on the root-to-winner branch path.

## Streaming response

With `stream: true`, the response media type is `text/event-stream`. Each event
contains an SSE `event:` line equal to the JSON payload's `type`, followed by one
`data:` line containing a single JSON object and a blank line. There is no
`[DONE]` sentinel. The `done` event is the terminal record and the HTTP stream
ends immediately after it.

### `branch_started`

```json
{"type":"branch_started","branch_id":"b0","parent_id":null}
```

- `branch_id`: unique branch identifier.
- `parent_id`: parent branch identifier, or `null` for a root branch.

### `token`

```json
{"type":"token","branch_id":"b0","token_index":0,"token":"hello","logprob":-0.25}
```

- `branch_id`: branch that owns this token span.
- `token_index`: zero-based index within that branch's own emitted tokens.
- `token`: exact text span contributed by this event.
- `logprob`: finite natural-log probability assigned by the engine's sampler to
  this sampled token. It MUST be the real sampled-token log probability, not a
  branch score or placeholder.

### `branch_pruned`

```json
{"type":"branch_pruned","branch_id":"b1","reason":"beam_pruned"}
```

- `branch_id`: branch being terminalized.
- `reason`: non-empty machine-readable or stable human-readable prune reason.

### `branch_merged`

```json
{"type":"branch_merged","branch_id":"b2","into_branch_id":"b0"}
```

- `branch_id`: branch being terminalized.
- `into_branch_id`: distinct, already-started, still-live merge target.

### `done`

```json
{
  "type": "done",
  "branch_id": "b0",
  "text": "hello",
  "finish_reason": "length",
  "usage": {
    "prompt_tokens": 1,
    "completion_tokens": 3,
    "total_tokens": 4
  },
  "counters": {
    "logical_tokens": 6,
    "physical_tokens": 3,
    "useful_tokens": 1,
    "elapsed_seconds": 0.01,
    "ttft_seconds": 0.001
  },
  "tree": {}
}
```

- `branch_id`: winning branch identifier.
- `text`: complete root-to-winner text.
- `finish_reason`: `stop` or `length`.
- `usage`: Usage object defined below.
- `counters`: non-negative `logical_tokens`, `physical_tokens`, and
  `useful_tokens`; positive `elapsed_seconds`; non-negative `ttft_seconds`.
- `tree`: required Tree Summary object.

## Shared objects

### Usage

`prompt_tokens`, `completion_tokens`, and `total_tokens` are non-negative
integers. `total_tokens` MUST equal `prompt_tokens + completion_tokens`.
`completion_tokens` MUST equal the total number of emitted `token` events across
all branches, not only the winning branch.

### Tree Summary

The `tree` object in both the non-stream response and `done` contains every field
below:

| Field | Type | Contract |
|---|---|---|
| `policy` | string | Executed tree policy. |
| `branch_count` | integer | Number of `branch_started` events; at least 1. |
| `pruned_count` | integer | Number of `branch_pruned` events. |
| `merged_count` | integer | Number of `branch_merged` events. |
| `winner_branch_id` | string | MUST equal `done.branch_id`. |
| `tokens_spent_per_branch` | object | Every branch ID mapped to its count of owned `token` events. |
| `final_scores` | object | Every branch ID mapped to its final branch score. This is branch-keyed, never positional. |
| `scorer` | string or null | Executed scorer identifier. |
| `kv_reuse_ratio` | number | `logical_tokens / physical_tokens`; finite and at least 1. |

For a served tree completion, `physical_tokens` MUST be positive and
`kv_reuse_ratio` MUST reconcile exactly with the `done.counters` values within
normal floating-point tolerance. This ratio is a multiplier, not a percentage:
`1` means no reuse, while `5` means five logical token references per physical
token stored.

## Stream invariants

1. A branch MUST be started exactly once before any event references it.
2. A non-null `parent_id` MUST reference an already-started branch. The ordered
   parent chain defines the branch's root-to-leaf `branch_path`.
3. `token_index` MUST start at 0 independently for each branch and increase by
   exactly 1. Tokens MUST NOT arrive after that branch is terminal.
4. Every started branch MUST receive exactly one terminal event: the winner gets
   `done`; every other branch gets `branch_pruned` or `branch_merged`.
5. `done` MUST occur exactly once, MUST be the final event, and the stream MUST
   not end without it.
6. `done.text` MUST equal the concatenation of `token` text across the winner's
   complete root-to-winner path. SDK exports MUST use that same lineage rule so
   forked branches retain shared-prefix tokens.
7. Usage, branch counts, prune/merge counts, per-branch token counts,
   `final_scores`, winner identity, and KV reuse MUST reconcile with the events
   and counters as defined above.

Any server engine that violates these invariants MUST fail the request rather
than serialize a dishonest stream. SDK consumers MUST reject a fully consumed
stream that violates them.
