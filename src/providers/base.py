from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from src.core.types import JsonDict


class ProviderError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProviderResult:
    final_text: str
    events: list[JsonDict]
    metadata: JsonDict


class Provider(Protocol):
    def run(
        self,
        prompt: str,
        *,
        artifacts_dir: Path,
        timeout_s: float,
        idle_timeout_s: float,
    ) -> ProviderResult: ...
