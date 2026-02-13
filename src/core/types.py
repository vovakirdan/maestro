from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

JsonDict = dict[str, Any]


class Status(StrEnum):
    OK = "OK"
    NEEDS_INPUT = "NEEDS_INPUT"
    FAILED = "FAILED"


@dataclass(frozen=True)
class StructuredReport:
    status: Status
    output: str
    next_inputs: str
    artifacts: tuple[str, ...] = ()
