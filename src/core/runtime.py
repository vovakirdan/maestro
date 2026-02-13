from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from src.core.types import JsonDict

ProviderType = Literal["deterministic", "codex_cli"]
PresetType = Literal["crt_v1"]
TaskKind = Literal["feature", "bug", "bootstrap"]


def orchestrator_root(workspace_dir: Path) -> Path:
    return workspace_dir / "orchestrator"


def generate_run_id() -> str:
    # Stable layout with unique IDs; includes UTC timestamp for sorting.
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    suffix = uuid.uuid4().hex[:8]
    return f"{ts}_{suffix}"


@dataclass(frozen=True)
class ProviderConfig:
    type: ProviderType
    command: tuple[str, ...] | None = None
    timeout_s: float = 120.0
    idle_timeout_s: float = 30.0

    def to_dict(self) -> JsonDict:
        d: JsonDict = {
            "type": self.type,
            "timeout_s": self.timeout_s,
            "idle_timeout_s": self.idle_timeout_s,
        }
        if self.command is not None:
            d["command"] = list(self.command)
        return d

    @staticmethod
    def from_dict(obj: dict[str, Any]) -> ProviderConfig:
        t = obj.get("type")
        if t not in ("deterministic", "codex_cli"):
            raise ValueError(f"provider.type must be 'deterministic' or 'codex_cli', got: {t!r}")
        command: tuple[str, ...] | None = None
        if t == "codex_cli":
            cmd = obj.get("command")
            if (
                not isinstance(cmd, list)
                or not cmd
                or not all(isinstance(x, str) and x for x in cmd)
            ):
                raise ValueError(
                    "provider.command must be a non-empty list of strings for codex_cli"
                )
            command = tuple(cmd)
        timeout_s = float(obj.get("timeout_s", 120.0))
        idle_timeout_s = float(obj.get("idle_timeout_s", 30.0))
        return ProviderConfig(
            type=t, command=command, timeout_s=timeout_s, idle_timeout_s=idle_timeout_s
        )


@dataclass(frozen=True)
class ActorConfig:
    actor_id: str
    packet_dir: str
    include_paths_in_prompt: bool = True

    def to_dict(self) -> JsonDict:
        return {
            "actor_id": self.actor_id,
            "packet_dir": self.packet_dir,
            "include_paths_in_prompt": self.include_paths_in_prompt,
        }

    @staticmethod
    def from_dict(obj: dict[str, Any]) -> ActorConfig:
        actor_id = obj.get("actor_id")
        packet_dir = obj.get("packet_dir")
        if not isinstance(actor_id, str) or not actor_id.strip():
            raise ValueError("actor.actor_id must be a non-empty string")
        if not isinstance(packet_dir, str) or not packet_dir.strip():
            raise ValueError("actor.packet_dir must be a non-empty string")
        include_paths = obj.get("include_paths_in_prompt", True)
        if not isinstance(include_paths, bool):
            raise ValueError("actor.include_paths_in_prompt must be boolean")
        return ActorConfig(
            actor_id=actor_id.strip(),
            packet_dir=packet_dir.strip(),
            include_paths_in_prompt=include_paths,
        )


@dataclass(frozen=True)
class OrchestrationConfig:
    """
    Optional run policy configuration.

    If omitted, the engine runs `actors` once, sequentially, in the order listed.
    """

    preset: PresetType
    max_returns: int = 3

    def to_dict(self) -> JsonDict:
        return {"preset": self.preset, "max_returns": self.max_returns}

    @staticmethod
    def from_dict(obj: dict[str, Any]) -> "OrchestrationConfig":
        preset = obj.get("preset")
        if preset == "crt":
            preset = "crt_v1"
        if preset not in ("crt_v1",):
            raise ValueError(f"orchestration.preset must be 'crt_v1', got: {preset!r}")
        max_returns_raw = obj.get("max_returns", 3)
        try:
            max_returns = int(max_returns_raw)
        except Exception as e:
            raise ValueError("orchestration.max_returns must be an int") from e
        if max_returns < 0:
            raise ValueError("orchestration.max_returns must be >= 0")
        return OrchestrationConfig(preset=preset, max_returns=max_returns)


@dataclass(frozen=True)
class TaskConfig:
    """
    Optional task "shape" configuration used to tailor packet templates and planning.

    `details_md` is user-provided supplemental context (markdown) captured at setup time.
    """

    kind: TaskKind
    details_md: str = ""

    def to_dict(self) -> JsonDict:
        d: JsonDict = {"kind": self.kind}
        if self.details_md.strip():
            d["details_md"] = self.details_md
        return d

    @staticmethod
    def from_dict(obj: dict[str, Any]) -> "TaskConfig":
        kind = obj.get("kind")
        if kind not in ("feature", "bug", "bootstrap"):
            raise ValueError(f"task.kind must be 'feature', 'bug', or 'bootstrap', got: {kind!r}")
        details_md = obj.get("details_md", "")
        if details_md is None:
            details_md = ""
        if not isinstance(details_md, str):
            raise ValueError("task.details_md must be a string")
        return TaskConfig(kind=kind, details_md=details_md)


@dataclass(frozen=True)
class PipelineConfig:
    version: int
    provider: ProviderConfig
    actors: tuple[ActorConfig, ...]
    orchestration: OrchestrationConfig | None = None
    goal: str | None = None
    task: TaskConfig | None = None

    def to_dict(self) -> JsonDict:
        d: JsonDict = {
            "version": self.version,
            "provider": self.provider.to_dict(),
            "actors": [a.to_dict() for a in self.actors],
        }
        if self.orchestration is not None:
            d["orchestration"] = self.orchestration.to_dict()
        if self.goal is not None and self.goal.strip():
            d["goal"] = self.goal
        if self.task is not None:
            d["task"] = self.task.to_dict()
        return d

    @staticmethod
    def from_dict(obj: dict[str, Any]) -> PipelineConfig:
        version = obj.get("version", 1)
        if not isinstance(version, int):
            raise ValueError("pipeline.version must be int")
        prov_raw = obj.get("provider")
        if not isinstance(prov_raw, dict):
            raise ValueError("pipeline.provider must be an object")
        provider = ProviderConfig.from_dict(prov_raw)
        actors_raw = obj.get("actors")
        if not isinstance(actors_raw, list) or not actors_raw:
            raise ValueError("pipeline.actors must be a non-empty list")
        actors = tuple(ActorConfig.from_dict(a) for a in actors_raw)
        orch_raw = obj.get("orchestration")
        orchestration: OrchestrationConfig | None = None
        if orch_raw is not None:
            if not isinstance(orch_raw, dict):
                raise ValueError("pipeline.orchestration must be an object")
            orchestration = OrchestrationConfig.from_dict(orch_raw)
        goal = obj.get("goal")
        if goal is not None and not isinstance(goal, str):
            raise ValueError("pipeline.goal must be a string")
        task_raw = obj.get("task")
        task: TaskConfig | None = None
        if task_raw is not None:
            if not isinstance(task_raw, dict):
                raise ValueError("pipeline.task must be an object")
            task = TaskConfig.from_dict(task_raw)
        return PipelineConfig(
            version=version,
            provider=provider,
            actors=actors,
            orchestration=orchestration,
            goal=goal,
            task=task,
        )


def load_pipeline(path: Path) -> PipelineConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("pipeline.json must contain a JSON object")
    return PipelineConfig.from_dict(raw)


def save_pipeline(path: Path, pipeline: PipelineConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(pipeline.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
