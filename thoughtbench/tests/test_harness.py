import json
from pathlib import Path

import httpx

from thoughtbench.cli import main
from thoughtbench.harness import (
    dry_run_requests,
    load_bench_tasks,
    load_harness_config,
    run_harness,
)
from thoughtbench.harness_schema import validate_benchmark_results


def test_bundled_task_sets_have_required_counts_and_gold_fields() -> None:
    root = Path(__file__).parents[1]
    gsm8k, _ = load_bench_tasks(root / "tasks" / "gsm8k_subset.jsonl")
    math12, _ = load_bench_tasks(root / "tasks" / "math12.jsonl")
    assert len(gsm8k) == 50
    assert len(math12) == 12
    assert all(task.gold for task in [*gsm8k, *math12])


def test_yaml_config_and_dry_run_print_exact_bodies_without_network(tmp_path, capsys) -> None:
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(json.dumps({"id": "one", "prompt": "2+2?", "gold": "4"}) + "\n", encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(
            [
                "label: dry",
                "engine_label: local",
                "model: model",
                "base_url: http://endpoint.test",
                "arm: single",
                "task_file: tasks.jsonl",
                "output_dir: results",
                "max_tokens: 8",
                "temperature: 0",
                "seeds: [7]",
                "concurrency: 1",
                "timeout: 5",
            ]
        ),
        encoding="utf-8",
    )
    loaded = load_harness_config(config)
    requests = dry_run_requests(loaded)
    assert requests[0]["body"] == {
        "model": "model",
        "messages": [{"role": "user", "content": "2+2?"}],
        "max_tokens": 8,
        "temperature": 0,
        "seed": 7,
    }

    assert main(["run", "--config", str(config), "--dry-run"]) == 0
    rendered = json.loads(capsys.readouterr().out)
    assert rendered == requests[0]


def test_full_harness_run_with_mocked_transport_writes_valid_results(tmp_path) -> None:
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(
        json.dumps({"id": "one", "prompt": "2+2?", "gold": "4"}) + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "label": "mock",
                "engine_label": "mock-engine",
                "model": "model",
                "base_url": "http://endpoint.test/v1",
                "arm": "best_of_n",
                "task_file": "tasks.jsonl",
                "output_dir": "results",
                "n": 3,
                "max_tokens": 8,
                "temperature": 0,
                "seeds": [7, 8],
                "concurrency": 1,
                "timeout": 5,
            }
        ),
        encoding="utf-8",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": "Answer: 4"}},
                    {"message": {"content": "Answer: 3"}},
                    {"message": {"content": "#### 4"}},
                ],
                "usage": {"completion_tokens": 12},
            },
        )

    results, output = run_harness(
        load_harness_config(config_path),
        transport=httpx.MockTransport(handler),
    )

    assert output.parent == tmp_path / "results"
    assert output.name.endswith("-mock.json")
    assert results.meta.base_url_redacted == "http://endpoint.test/v1"
    assert results.summary.accuracy == 1
    assert results.summary.mean_tokens == 12
    assert results.summary.tokens_per_correct == 12
    assert len(results.tasks) == 2
    validate_benchmark_results(json.loads(output.read_text(encoding="utf-8")))
