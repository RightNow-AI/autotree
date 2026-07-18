#!/usr/bin/env bash

set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_modeling=false
results=()

if [[ "${1:-}" == "--modeling" ]]; then
  run_modeling=true
  shift
fi

if [[ $# -ne 0 ]]; then
  echo "usage: scripts/verify-local.sh [--modeling]" >&2
  exit 2
fi

show_summary() {
  echo
  echo "Verification summary"
  echo "--------------------"
  local result
  for result in "${results[@]}"; do
    printf '%-32s %s\n' "${result%%|*}" "${result#*|}"
  done
}

run_gate() {
  local name="$1"
  shift
  echo
  echo "==> $name"
  if "$@"; then
    results+=("$name|PASS")
  else
    local status=$?
    results+=("$name|FAIL")
    show_summary
    exit "$status"
  fi
}

core_venv() { (cd "$repo_root/core" && uv venv --python 3.12 --clear); }
core_install() { (cd "$repo_root/core" && uv pip install -e '.[dev]'); }
core_tests() { (cd "$repo_root/core" && uv run --no-sync pytest -q tests/kv tests/kernels); }
scheduler_fmt() { (cd "$repo_root/scheduler" && cargo fmt --check); }
scheduler_clippy() { (cd "$repo_root/scheduler" && cargo clippy --all-targets -- -D warnings); }
scheduler_test() { (cd "$repo_root/scheduler" && cargo test); }
scheduler_python() { (cd "$repo_root/scheduler" && cargo check --features python); }
serve_venv() { (cd "$repo_root/serve" && uv venv --python 3.12 --clear); }
serve_install() { (cd "$repo_root/serve" && uv pip install -e '.[dev]'); }
serve_tests() { (cd "$repo_root/serve" && uv run --no-sync pytest -q); }
sdk_venv() { (cd "$repo_root/sdk" && uv venv --python 3.12 --clear); }
sdk_install() { (cd "$repo_root/sdk" && uv pip install -e . -e ../serve 'hypothesis>=6.100,<7' 'pytest>=8.2,<9'); }
sdk_tests() { (cd "$repo_root/sdk" && uv run --no-sync pytest -q --ignore=tests/test_server_contract.py); }
wire_contract_test() { (cd "$repo_root/sdk" && uv run --no-sync pytest -q tests/test_server_contract.py); }
thoughtbench_venv() { (cd "$repo_root/thoughtbench" && uv venv --python 3.12 --clear); }
thoughtbench_install() { (cd "$repo_root/thoughtbench" && uv pip install -e . -e ../sdk -e ../serve 'httpx>=0.28,<1' 'pytest>=8.3,<9' 'uvicorn>=0.34,<1'); }
thoughtbench_tests() { (cd "$repo_root/thoughtbench" && uv run --no-sync pytest -q); }
yaml_parse() {
  cd "$repo_root" && uv run --with pyyaml python -c "import pathlib,yaml; files=[pathlib.Path('.github/workflows/ci.yml'),pathlib.Path('.github/workflows/gpu-parity.yml')]; [yaml.safe_load(path.read_text(encoding='utf-8')) for path in files]; print('Parsed workflow YAML:', ', '.join(map(str, files)))"
}
modeling_install() { (cd "$repo_root/core" && uv pip install -e '.[dev,modeling]'); }
modeling_tests() { (cd "$repo_root/core" && HF_HOME="${TMPDIR:-/tmp}/autotree-huggingface" uv run --no-sync pytest -q tests); }

run_gate "core: create Python 3.12 env" core_venv
run_gate "core: install dev dependencies" core_install
run_gate "core: KV + kernel tests" core_tests
run_gate "scheduler: fmt" scheduler_fmt
run_gate "scheduler: clippy" scheduler_clippy
run_gate "scheduler: test" scheduler_test
run_gate "scheduler: python feature" scheduler_python

if [[ -d "$repo_root/serve" ]]; then
  run_gate "serve: create Python 3.12 env" serve_venv
  run_gate "serve: install dev dependencies" serve_install
  run_gate "serve: tests" serve_tests
else
  echo
  echo "==> serve: tests"
  echo "serve/ is absent; skipping the serve gate until that package merges."
  results+=("serve: tests|SKIP (serve/ absent)")
fi

run_gate "sdk: create Python 3.12 env" sdk_venv
run_gate "sdk: install test dependencies" sdk_install
run_gate "sdk: unit tests" sdk_tests
run_gate "wire: real serve + SDK contract" wire_contract_test
run_gate "thoughtbench: create Python 3.12 env" thoughtbench_venv
run_gate "thoughtbench: install test dependencies" thoughtbench_install
run_gate "thoughtbench: tests" thoughtbench_tests

run_gate "workflow YAML parse" yaml_parse

if [[ "$run_modeling" == true ]]; then
  run_gate "modeling: install dependencies" modeling_install
  run_gate "modeling: core tests" modeling_tests
else
  results+=("modeling: scheduled/manual|SKIP (use --modeling)")
fi

show_summary
