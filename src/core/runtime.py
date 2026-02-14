from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from src.core.types import JsonDict

ProviderType = Literal["deterministic", "codex_cli", "gemini_cli", "claude_cli"]
PresetType = Literal["crt_v1", "cr_v1", "linear_v1"]
TaskKind = Literal["feature", "bug", "bootstrap", "other"]
SeverityTag = Literal["BLOCKER", "MAJOR", "MINOR", "NIT"]
ReviewTrigger = Literal["status_failed", "any_tag"]
ReviewAction = Literal["return", "escalate"]


@dataclass(frozen=True)
class ReviewReactionPolicy:
    """
    Optional orchestration rule for reacting to a specific actor's review output.

    This is intended for "reviewer"-like steps that produce findings tagged with:
        [BLOCKER], [MAJOR], [MINOR], [NIT]
    The orchestrator can use this to decide whether to return to an implementer
    (return_to) or escalate for human input.
    """

    actor_id: str
    trigger: ReviewTrigger = "status_failed"
    min_severity: SeverityTag | None = None
    max_severity: SeverityTag | None = None
    max_blockers: int | None = None
    max_majors: int | None = None
    max_total: int | None = None
    on_trigger: ReviewAction = "return"
    on_overflow: ReviewAction = "escalate"

    def to_dict(self) -> JsonDict:
        d: JsonDict = {
            "actor_id": self.actor_id,
            "trigger": self.trigger,
            "on_trigger": self.on_trigger,
            "on_overflow": self.on_overflow,
        }
        if self.min_severity is not None:
            d["min_severity"] = self.min_severity
        if self.max_severity is not None:
            d["max_severity"] = self.max_severity
        if self.max_blockers is not None:
            d["max_blockers"] = self.max_blockers
        if self.max_majors is not None:
            d["max_majors"] = self.max_majors
        if self.max_total is not None:
            d["max_total"] = self.max_total
        return d

    @staticmethod
    def from_dict(obj: dict[str, Any]) -> "ReviewReactionPolicy":
        actor_id = obj.get("actor_id")
        if not isinstance(actor_id, str) or not actor_id.strip():
            raise ValueError("review_policy.actor_id must be a non-empty string")
        trigger = obj.get("trigger", "status_failed")
        if trigger not in ("status_failed", "any_tag"):
            raise ValueError("review_policy.trigger must be 'status_failed' or 'any_tag'")
        on_trigger = obj.get("on_trigger", "return")
        if on_trigger not in ("return", "escalate"):
            raise ValueError("review_policy.on_trigger must be 'return' or 'escalate'")
        on_overflow = obj.get("on_overflow", "escalate")
        if on_overflow not in ("return", "escalate"):
            raise ValueError("review_policy.on_overflow must be 'return' or 'escalate'")

        min_sev = obj.get("min_severity")
        if min_sev is not None and min_sev not in ("BLOCKER", "MAJOR", "MINOR", "NIT"):
            raise ValueError("review_policy.min_severity must be one of BLOCKER/MAJOR/MINOR/NIT")
        max_sev = obj.get("max_severity")
        if max_sev is not None and max_sev not in ("BLOCKER", "MAJOR", "MINOR", "NIT"):
            raise ValueError("review_policy.max_severity must be one of BLOCKER/MAJOR/MINOR/NIT")

        def _opt_int(name: str) -> int | None:
            raw = obj.get(name)
            if raw is None:
                return None
            try:
                v = int(raw)
            except Exception as e:
                raise ValueError(f"review_policy.{name} must be an int") from e
            if v < 0:
                raise ValueError(f"review_policy.{name} must be >= 0")
            return v

        return ReviewReactionPolicy(
            actor_id=actor_id.strip(),
            trigger=trigger,
            min_severity=min_sev,
            max_severity=max_sev,
            max_blockers=_opt_int("max_blockers"),
            max_majors=_opt_int("max_majors"),
            max_total=_opt_int("max_total"),
            on_trigger=on_trigger,
            on_overflow=on_overflow,
        )


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
        if t not in ("deterministic", "codex_cli", "gemini_cli", "claude_cli"):
            raise ValueError(
                "provider.type must be one of: "
                "'deterministic', 'codex_cli', 'gemini_cli', 'claude_cli', got: "
                f"{t!r}"
            )
        command: tuple[str, ...] | None = None
        if t in {"codex_cli", "gemini_cli", "claude_cli"}:
            cmd = obj.get("command")
            if (
                not isinstance(cmd, list)
                or not cmd
                or not all(isinstance(x, str) and x for x in cmd)
            ):
                raise ValueError(
                    "provider.command must be a non-empty list of strings for "
                    "codex_cli/gemini_cli/claude_cli"
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
    provider: ProviderConfig | None = None

    def to_dict(self) -> JsonDict:
        d: JsonDict = {
            "actor_id": self.actor_id,
            "packet_dir": self.packet_dir,
            "include_paths_in_prompt": self.include_paths_in_prompt,
        }
        if self.provider is not None:
            d["provider"] = self.provider.to_dict()
        return d

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

        provider_raw = obj.get("provider")
        provider: ProviderConfig | None = None
        if provider_raw is not None:
            if not isinstance(provider_raw, dict):
                raise ValueError("actor.provider must be an object")
            provider = ProviderConfig.from_dict(provider_raw)
        return ActorConfig(
            actor_id=actor_id.strip(),
            packet_dir=packet_dir.strip(),
            include_paths_in_prompt=include_paths,
            provider=provider,
        )


@dataclass(frozen=True)
class OrchestrationConfig:
    """
    Optional run policy configuration.

    If omitted, the engine runs `actors` once, sequentially, in the order listed.
    """

    preset: PresetType
    max_returns: int = 3
    return_to: str | None = None
    return_from: tuple[str, ...] = ()
    review_policies: tuple[ReviewReactionPolicy, ...] = ()

    def to_dict(self) -> JsonDict:
        d: JsonDict = {"preset": self.preset, "max_returns": self.max_returns}
        if self.return_to is not None and self.return_to.strip():
            d["return_to"] = self.return_to
        if self.return_from:
            d["return_from"] = list(self.return_from)
        if self.review_policies:
            d["review_policies"] = [p.to_dict() for p in self.review_policies]
        return d

    @staticmethod
    def from_dict(obj: dict[str, Any]) -> "OrchestrationConfig":
        preset = obj.get("preset")
        if preset == "crt":
            preset = "crt_v1"
        if preset == "cr":
            preset = "cr_v1"
        if preset == "linear":
            preset = "linear_v1"
        if preset not in ("crt_v1", "cr_v1", "linear_v1"):
            raise ValueError(
                "orchestration.preset must be 'crt_v1', 'cr_v1', or 'linear_v1', got: "
                f"{preset!r}"
            )
        max_returns_raw = obj.get("max_returns", 3)
        try:
            max_returns = int(max_returns_raw)
        except Exception as e:
            raise ValueError("orchestration.max_returns must be an int") from e
        if max_returns < 0:
            raise ValueError("orchestration.max_returns must be >= 0")

        return_to_raw = obj.get("return_to")
        return_to = None
        if return_to_raw is not None:
            if not isinstance(return_to_raw, str):
                raise ValueError("orchestration.return_to must be a string")
            return_to = return_to_raw.strip() or None

        return_from_raw = obj.get("return_from")
        return_from: tuple[str, ...] = ()
        if return_from_raw is not None:
            if (
                not isinstance(return_from_raw, list)
                or not all(isinstance(x, str) and x.strip() for x in return_from_raw)
            ):
                raise ValueError("orchestration.return_from must be a list of strings")
            return_from = tuple(x.strip() for x in return_from_raw)

        review_raw = obj.get("review_policies")
        review_policies: tuple[ReviewReactionPolicy, ...] = ()
        if review_raw is not None:
            if not isinstance(review_raw, list) or not all(isinstance(x, dict) for x in review_raw):
                raise ValueError("orchestration.review_policies must be a list of objects")
            parsed = [ReviewReactionPolicy.from_dict(x) for x in review_raw]
            seen: set[str] = set()
            for p in parsed:
                if p.actor_id in seen:
                    raise ValueError(
                        "orchestration.review_policies actor_id must be unique; "
                        f"duplicate: {p.actor_id!r}"
                    )
                seen.add(p.actor_id)
            review_policies = tuple(parsed)

        # Preset defaults (backwards-compatible).
        if preset == "crt_v1":
            if return_to is None:
                return_to = "coder"
            if not return_from:
                return_from = ("reviewer", "tester")
        elif preset == "cr_v1":
            if return_to is None:
                return_to = "coder"
            if not return_from:
                return_from = ("reviewer",)

        return OrchestrationConfig(
            preset=preset,
            max_returns=max_returns,
            return_to=return_to,
            return_from=return_from,
            review_policies=review_policies,
        )


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
        if kind not in ("feature", "bug", "bootstrap", "other"):
            raise ValueError(
                "task.kind must be 'feature', 'bug', 'bootstrap', or 'other', got: " f"{kind!r}"
            )
        details_md = obj.get("details_md", "")
        if details_md is None:
            details_md = ""
        if not isinstance(details_md, str):
            raise ValueError("task.details_md must be a string")
        return TaskConfig(kind=kind, details_md=details_md)


GitMode = Literal["off", "check", "branch"]


@dataclass(frozen=True)
class GitDefaultsConfig:
    """
    Optional defaults used by the CLI and engine git policy.

    This does not execute any git actions by itself; it only provides defaults.
    """

    mode: GitMode = "off"
    branch_prefix: str = "orch/"
    branch_template: str | None = None
    branch_slug: str | None = None
    auto_commit: bool = False
    commit_message_template: str | None = None

    def to_dict(self) -> JsonDict:
        d: JsonDict = {
            "mode": self.mode,
            "branch_prefix": self.branch_prefix,
            "auto_commit": self.auto_commit,
        }
        if self.branch_template is not None and self.branch_template.strip():
            d["branch_template"] = self.branch_template
        if self.branch_slug is not None and self.branch_slug.strip():
            d["branch_slug"] = self.branch_slug
        if self.commit_message_template is not None and self.commit_message_template.strip():
            d["commit_message_template"] = self.commit_message_template
        return d

    @staticmethod
    def from_dict(obj: dict[str, Any]) -> "GitDefaultsConfig":
        mode = obj.get("mode", "off")
        if mode not in ("off", "check", "branch"):
            raise ValueError("git_defaults.mode must be 'off', 'check', or 'branch'")
        branch_prefix = obj.get("branch_prefix", "orch/")
        if not isinstance(branch_prefix, str):
            raise ValueError("git_defaults.branch_prefix must be a string")
        auto_commit = obj.get("auto_commit", False)
        if not isinstance(auto_commit, bool):
            raise ValueError("git_defaults.auto_commit must be boolean")
        branch_template = obj.get("branch_template")
        if branch_template is not None and not isinstance(branch_template, str):
            raise ValueError("git_defaults.branch_template must be a string")
        branch_slug = obj.get("branch_slug")
        if branch_slug is not None and not isinstance(branch_slug, str):
            raise ValueError("git_defaults.branch_slug must be a string")
        commit_message_template = obj.get("commit_message_template")
        if commit_message_template is not None and not isinstance(commit_message_template, str):
            raise ValueError("git_defaults.commit_message_template must be a string")
        return GitDefaultsConfig(
            mode=mode,
            branch_prefix=branch_prefix.strip() or "orch/",
            branch_template=branch_template.strip() if isinstance(branch_template, str) else None,
            branch_slug=branch_slug.strip() if isinstance(branch_slug, str) else None,
            auto_commit=auto_commit,
            commit_message_template=commit_message_template.strip()
            if isinstance(commit_message_template, str)
            else None,
        )


MRMode = Literal["off", "instructions"]


@dataclass(frozen=True)
class DeliveryConfig:
    """
    Optional delivery settings (e.g., merge request preparation).

    This tool does not assume a specific host (GitHub/GitLab); it can emit
    human-readable instructions at the end of a run.
    """

    mr_mode: MRMode = "off"
    remote: str = "origin"
    target_branch: str = "main"
    title_template: str | None = None
    body_template: str | None = None

    def to_dict(self) -> JsonDict:
        d: JsonDict = {
            "mr_mode": self.mr_mode,
            "remote": self.remote,
            "target_branch": self.target_branch,
        }
        if self.title_template is not None and self.title_template.strip():
            d["title_template"] = self.title_template
        if self.body_template is not None and self.body_template.strip():
            d["body_template"] = self.body_template
        return d

    @staticmethod
    def from_dict(obj: dict[str, Any]) -> "DeliveryConfig":
        mr_mode = obj.get("mr_mode", "off")
        if mr_mode not in ("off", "instructions"):
            raise ValueError("delivery.mr_mode must be 'off' or 'instructions'")
        remote = obj.get("remote", "origin")
        target_branch = obj.get("target_branch", "main")
        if not isinstance(remote, str):
            raise ValueError("delivery.remote must be a string")
        if not isinstance(target_branch, str):
            raise ValueError("delivery.target_branch must be a string")
        title_template = obj.get("title_template")
        if title_template is not None and not isinstance(title_template, str):
            raise ValueError("delivery.title_template must be a string")
        body_template = obj.get("body_template")
        if body_template is not None and not isinstance(body_template, str):
            raise ValueError("delivery.body_template must be a string")
        return DeliveryConfig(
            mr_mode=mr_mode,
            remote=remote.strip() or "origin",
            target_branch=target_branch.strip() or "main",
            title_template=title_template.strip() if isinstance(title_template, str) else None,
            body_template=body_template.strip() if isinstance(body_template, str) else None,
        )


@dataclass(frozen=True)
class PipelineConfig:
    version: int
    provider: ProviderConfig
    actors: tuple[ActorConfig, ...]
    orchestration: OrchestrationConfig | None = None
    goal: str | None = None
    task: TaskConfig | None = None
    git_defaults: GitDefaultsConfig | None = None
    delivery: DeliveryConfig | None = None

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
        if self.git_defaults is not None:
            d["git_defaults"] = self.git_defaults.to_dict()
        if self.delivery is not None:
            d["delivery"] = self.delivery.to_dict()
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
        git_raw = obj.get("git_defaults")
        git_defaults: GitDefaultsConfig | None = None
        if git_raw is not None:
            if not isinstance(git_raw, dict):
                raise ValueError("pipeline.git_defaults must be an object")
            git_defaults = GitDefaultsConfig.from_dict(git_raw)
        delivery_raw = obj.get("delivery")
        delivery: DeliveryConfig | None = None
        if delivery_raw is not None:
            if not isinstance(delivery_raw, dict):
                raise ValueError("pipeline.delivery must be an object")
            delivery = DeliveryConfig.from_dict(delivery_raw)
        return PipelineConfig(
            version=version,
            provider=provider,
            actors=actors,
            orchestration=orchestration,
            goal=goal,
            task=task,
            git_defaults=git_defaults,
            delivery=delivery,
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
