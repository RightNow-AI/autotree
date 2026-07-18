import json

import pytest
from jsonschema import ValidationError

from thoughtbench.models import RESULTS_SCHEMA_VERSION
from thoughtbench.schema import results_json_schema, validate_results_payload


def test_results_schema_is_versioned_and_forbids_unknown_top_level_fields() -> None:
    schema = results_json_schema()

    assert schema["$id"].endswith(RESULTS_SCHEMA_VERSION)
    assert schema["additionalProperties"] is False


def test_schema_rejects_a_payload_without_fixture_honesty_stamp() -> None:
    with pytest.raises(ValidationError):
        validate_results_payload(
            {
                "schema_version": RESULTS_SCHEMA_VERSION,
                "benchmark_claims_allowed": False,
            }
        )


def test_schema_can_be_serialized_for_downstream_consumers() -> None:
    rendered = json.dumps(results_json_schema())

    assert "thoughtbench.results.v2" in rendered
    assert "artifact_notice" in rendered
