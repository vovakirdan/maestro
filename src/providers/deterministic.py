from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from src.core.types import JsonDict
from src.providers.base import ProviderResult


def _extract_section(prompt: str, *, section: str) -> str:
    """
    Extracts the text content of a rendered Packet section:
        === SECTION ===
        [optional SOURCE: ...]
        <content...>
        === NEXT ===
    """
    lines = prompt.splitlines()
    header = f"=== {section} ==="
    start = None
    for i, line in enumerate(lines):
        if line.strip() == header:
            start = i + 1
            break
    if start is None:
        return ""

    if start < len(lines) and lines[start].startswith("SOURCE:"):
        start += 1

    out: list[str] = []
    for line in lines[start:]:
        s = line.strip()
        if s.startswith("=== ") and s.endswith(" ==="):
            break
        out.append(line)
    return "\n".join(out).strip() + ("\n" if out else "")


@dataclass(frozen=True)
class DeterministicProvider:
    """Offline provider for deterministic runs (no network, no subprocess)."""

    provider_name: str = "deterministic"

    def run(
        self,
        prompt: str,
        *,
        artifacts_dir: Path,
        timeout_s: float,
        idle_timeout_s: float,
        on_event: Callable[[JsonDict], None] | None = None,
    ) -> ProviderResult:
        started_monotonic = time.monotonic()
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        events: list[JsonDict] = []

        def emit(ev: JsonDict) -> None:
            events.append(ev)
            if on_event is None:
                return
            try:
                on_event(ev)
            except Exception:
                # Never fail a run due to UI/event-consumer issues.
                pass

        emit({"type": "provider", "provider": self.provider_name, "event": "start"})

        prompt_bytes = prompt.encode("utf-8", errors="strict")
        prompt_sha256 = hashlib.sha256(prompt_bytes).hexdigest()
        inputs_text = _extract_section(prompt, section="INPUTS")
        inputs_sha256 = hashlib.sha256(inputs_text.encode("utf-8", errors="strict")).hexdigest()

        emit(
            {
                "type": "provider",
                "provider": self.provider_name,
                "event": "prompt_digest",
                "sha256": prompt_sha256,
            }
        )

        final_obj: JsonDict = {
            "status": "OK",
            "output": (
                "DeterministicProvider: no AI provider configured.\n"
                f"prompt_sha256={prompt_sha256}\n"
                f"inputs_sha256={inputs_sha256}\n"
            ),
            "next_inputs": "",
        }
        final_text = json.dumps(final_obj, indent=2, sort_keys=True) + "\n"

        emit({"type": "provider", "provider": self.provider_name, "event": "finish"})
        metadata: JsonDict = {
            "provider": self.provider_name,
            "duration_s": round(time.monotonic() - started_monotonic, 6),
            "timeout_s": timeout_s,
            "idle_timeout_s": idle_timeout_s,
            "prompt_sha256": prompt_sha256,
            "prompt_bytes": len(prompt_bytes),
            "inputs_sha256": inputs_sha256,
        }

        (artifacts_dir / "provider_final.txt").write_text(final_text, encoding="utf-8")
        (artifacts_dir / "provider_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with (artifacts_dir / "provider_events.jsonl").open("w", encoding="utf-8") as f:
            for ev in events:
                f.write(json.dumps(ev, sort_keys=True) + "\n")

        return ProviderResult(final_text=final_text, events=events, metadata=metadata)
