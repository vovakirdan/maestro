from __future__ import annotations

import json
from dataclasses import dataclass

from src.core.types import JsonDict, Status, StructuredReport

FORMAT_SENTINEL = "FORMAT: ORCH_JSON_V1"


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    errors: tuple[str, ...]
    report: StructuredReport | None = None
    raw_object: JsonDict | None = None


def _extract_first_json_object(text: str) -> tuple[JsonDict | None, str | None]:
    decoder = json.JSONDecoder()

    # Try from each '{' occurrence; robust to prefixes like code fences or log lines.
    idx = text.find("{")
    while idx != -1:
        try:
            obj, _end = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            idx = text.find("{", idx + 1)
            continue
        if isinstance(obj, dict):
            # mypy: obj is dict[str, Any] logically, but JSONDecoder returns Any.
            return obj, None
        return None, "Parsed JSON is not an object"
    return None, "No JSON object found in output"


class ReportValidator:
    """
    Strict validator for ORCH_JSON_V1.

    If REPORT_FORMAT.md does not contain the sentinel, this still attempts ORCH_JSON_V1
    (so the system remains minimal), but reports the mismatch as an error to help users.
    """

    def validate(self, *, report_format_text: str, output_text: str) -> ValidationResult:
        errors: list[str] = []
        if FORMAT_SENTINEL not in report_format_text:
            errors.append(f"REPORT_FORMAT missing sentinel {FORMAT_SENTINEL!r}")

        obj, err = _extract_first_json_object(output_text)
        if err:
            errors.append(err)
            return ValidationResult(ok=False, errors=tuple(errors), report=None, raw_object=None)
        assert obj is not None

        allowed_keys = {"status", "output", "next_inputs", "artifacts"}
        required_keys = {"status", "output", "next_inputs"}
        extra_keys = set(obj.keys()) - allowed_keys
        missing_keys = required_keys - set(obj.keys())
        if extra_keys:
            errors.append(f"Unexpected keys: {sorted(extra_keys)}")
        if missing_keys:
            errors.append(f"Missing keys: {sorted(missing_keys)}")

        status_raw = obj.get("status")
        if not isinstance(status_raw, str):
            errors.append("status must be a string")
            status = None
        else:
            try:
                status = Status(status_raw)
            except ValueError:
                errors.append(
                    f"status must be one of {[s.value for s in Status]}, got: {status_raw!r}"
                )
                status = None

        output = obj.get("output")
        if not isinstance(output, str):
            errors.append("output must be a string")

        next_inputs = obj.get("next_inputs")
        if not isinstance(next_inputs, str):
            errors.append("next_inputs must be a string")

        artifacts_tuple: tuple[str, ...] = ()
        if "artifacts" in obj:
            artifacts = obj.get("artifacts")
            if isinstance(artifacts, list) and all(isinstance(x, str) for x in artifacts):
                artifacts_tuple = tuple(artifacts)
            else:
                errors.append("artifacts must be a list of strings")

        if errors:
            return ValidationResult(ok=False, errors=tuple(errors), report=None, raw_object=obj)

        assert status is not None
        assert isinstance(output, str)
        assert isinstance(next_inputs, str)
        report = StructuredReport(
            status=status, output=output, next_inputs=next_inputs, artifacts=artifacts_tuple
        )
        return ValidationResult(ok=True, errors=tuple(), report=report, raw_object=obj)


def retry_instructions(errors: tuple[str, ...]) -> str:
    joined = "\n".join(f"- {e}" for e in errors)
    return (
        "Your previous response did not match REPORT_FORMAT.\n"
        "Return exactly one JSON object matching ORCH_JSON_V1.\n"
        "Do not wrap it in markdown fences.\n"
        "Validation errors:\n"
        f"{joined}\n"
    )
