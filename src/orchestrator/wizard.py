from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from src.core.runtime import DeliveryConfig, GitDefaultsConfig, OrchestrationConfig, TaskConfig
from src.core.types import JsonDict
from src.orchestrator.spec import AgentSpec
from src.providers.base import Provider


def _extract_first_json_object(text: str) -> JsonDict | None:
    decoder = json.JSONDecoder()
    idx = text.find("{")
    while idx != -1:
        try:
            obj, _end = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            idx = text.find("{", idx + 1)
            continue
        return obj if isinstance(obj, dict) else None
    return None


_RE_SLUG_SAFE = re.compile(r"[^a-z0-9._/-]+")


def sanitize_slug(value: str) -> str:
    """
    Convert arbitrary text into a short ASCII slug suitable for branch names.
    """
    v = value.strip().lower()
    v = v.replace(" ", "-")
    v = _RE_SLUG_SAFE.sub("-", v)
    v = re.sub(r"-{2,}", "-", v).strip("-")
    return v[:64] if v else "task"


@dataclass(frozen=True)
class WizardAnalyzeResult:
    brief_md: str
    git: GitDefaultsConfig | None = None
    delivery: DeliveryConfig | None = None
    orchestration: OrchestrationConfig | None = None


def run_setup_wizard_analyze(
    provider: Provider,
    *,
    workspace_dir: Path,
    goal: str,
    task: TaskConfig | None,
    agents: tuple[AgentSpec, ...],
    orchestration: OrchestrationConfig | None,
    git_defaults: GitDefaultsConfig | None,
    delivery: DeliveryConfig | None,
    interaction_notes: str,
    artifacts_dir: Path,
    timeout_s: float,
    idle_timeout_s: float,
    on_event: Callable[[JsonDict], None] | None = None,
) -> tuple[WizardAnalyzeResult | None, JsonDict]:
    """
    Setup wizard (AI): analyze the workspace + goal and produce:
    - a high-signal brief/decomposition (markdown)
    - optional suggested git defaults (branch slug/template, commit message template)
    - optional MR instructions templates
    - optional orchestration override (linear returns config)

    Returns (result_or_none, provider_metadata).
    """
    task_kind = task.kind if task is not None else "other"
    task_details = task.details_md.strip() if task is not None else ""
    details_block = f"Task details:\n{task_details}\n\n" if task_details else ""

    # Keep this wizard read-only.
    prompt = "".join(
        [
            "You are a setup wizard for a local multi-agent orchestrator.\n\n",
            "Your job is to prepare a high-quality setup brief and configuration suggestions.\n\n",
            "Hard constraints:\n",
            "- Read-only workspace inspection: do not modify files.\n",
            "- Do not run destructive commands (no rm -rf, no installs).\n",
            "- Do not mention agents/pipeline/orchestration in the brief; the brief is for a human and for packet docs.\n",
            "- Use ASCII only.\n",
            "- Do not output markdown fences.\n\n",
            "Workspace:\n",
            f"- root: {workspace_dir.as_posix()}\n\n",
            f"Task type: {task_kind}\n",
            details_block,
            "Goal:\n",
            f"{goal.strip()}\n\n",
            "Selected roles (in current intended order):\n",
        ]
    )
    for a in agents:
        spec = a.specialization.strip() or "(none)"
        prompt += f"- {a.actor_id}: template={a.template_id} specialization={spec}\n"
    prompt += "\n"

    if orchestration is not None:
        prompt += (
            "Current orchestration intent (may be improved, but keep it linear):\n"
            f"- preset: {orchestration.preset}\n"
            f"- max_returns: {orchestration.max_returns}\n"
            f"- return_to: {orchestration.return_to or '(none)'}\n"
            f"- return_from: {', '.join(orchestration.return_from) if orchestration.return_from else '(none)'}\n\n"
        )

    if interaction_notes.strip():
        prompt += "User workflow/interaction notes:\n" + interaction_notes.strip() + "\n\n"

    if git_defaults is not None:
        prompt += (
            "Git defaults (you may suggest improved templates/slugs):\n"
            f"- mode: {git_defaults.mode}\n"
            f"- branch_prefix: {git_defaults.branch_prefix}\n"
            f"- branch_template: {git_defaults.branch_template or '(none)'}\n"
            f"- branch_slug: {git_defaults.branch_slug or '(none)'}\n"
            f"- auto_commit: {git_defaults.auto_commit}\n"
            f"- commit_message_template: {git_defaults.commit_message_template or '(none)'}\n\n"
        )

    if delivery is not None:
        prompt += (
            "Delivery (merge request) config:\n"
            f"- mr_mode: {delivery.mr_mode}\n"
            f"- remote: {delivery.remote}\n"
            f"- target_branch: {delivery.target_branch}\n"
            f"- title_template: {delivery.title_template or '(none)'}\n"
            f"- body_template: {delivery.body_template or '(none)'}\n\n"
        )

    prompt += (
        "Output:\n"
        "Return exactly one JSON object with keys:\n"
        "- brief_md (markdown string)\n"
        "- branch_slug (string, optional; short ASCII slug)\n"
        "- branch_template (string, optional; can use placeholders {prefix}, {slug}, {run_id}, {date_utc})\n"
        "- commit_message_template (string, optional; placeholders {run_id}, {step}, {actor_id}, {label})\n"
        "- mr_title_template (string, optional)\n"
        "- mr_body_template (string, optional)\n"
        "- return_to (string, optional)\n"
        "- return_from (list of strings, optional)\n"
        "- max_returns (int, optional)\n"
        "No extra keys. Strings must be non-empty when present.\n"
    )

    res = provider.run(
        prompt,
        artifacts_dir=artifacts_dir,
        timeout_s=timeout_s,
        idle_timeout_s=idle_timeout_s,
        on_event=on_event,
    )

    obj = _extract_first_json_object(res.final_text)
    if not isinstance(obj, dict) or "brief_md" not in obj:
        return None, res.metadata
    brief_md = obj.get("brief_md")
    if not isinstance(brief_md, str) or not brief_md.strip():
        return None, res.metadata

    # Best-effort extract optional config.
    branch_slug = obj.get("branch_slug")
    branch_template = obj.get("branch_template")
    commit_message_template = obj.get("commit_message_template")
    mr_title_template = obj.get("mr_title_template")
    mr_body_template = obj.get("mr_body_template")

    out_git = git_defaults
    if any(isinstance(x, str) and x.strip() for x in (branch_slug, branch_template, commit_message_template)):
        prefix = git_defaults.branch_prefix if git_defaults is not None else "orch/"
        mode = git_defaults.mode if git_defaults is not None else "branch"
        auto_commit = git_defaults.auto_commit if git_defaults is not None else False
        out_git = GitDefaultsConfig(
            mode=mode,
            branch_prefix=prefix,
            branch_template=branch_template.strip()
            if isinstance(branch_template, str) and branch_template.strip()
            else (git_defaults.branch_template if git_defaults is not None else None),
            branch_slug=sanitize_slug(branch_slug)
            if isinstance(branch_slug, str) and branch_slug.strip()
            else (git_defaults.branch_slug if git_defaults is not None else None),
            auto_commit=auto_commit,
            commit_message_template=commit_message_template.strip()
            if isinstance(commit_message_template, str) and commit_message_template.strip()
            else (git_defaults.commit_message_template if git_defaults is not None else None),
        )

    out_delivery = delivery
    if any(isinstance(x, str) and x.strip() for x in (mr_title_template, mr_body_template)):
        mr_mode = delivery.mr_mode if delivery is not None else "instructions"
        remote = delivery.remote if delivery is not None else "origin"
        target_branch = delivery.target_branch if delivery is not None else "main"
        out_delivery = DeliveryConfig(
            mr_mode=mr_mode,
            remote=remote,
            target_branch=target_branch,
            title_template=mr_title_template.strip()
            if isinstance(mr_title_template, str) and mr_title_template.strip()
            else (delivery.title_template if delivery is not None else None),
            body_template=mr_body_template.strip()
            if isinstance(mr_body_template, str) and mr_body_template.strip()
            else (delivery.body_template if delivery is not None else None),
        )

    out_orch = orchestration
    if orchestration is not None:
        return_to = obj.get("return_to")
        return_from = obj.get("return_from")
        max_returns = obj.get("max_returns")
        try:
            next_max = orchestration.max_returns
            if isinstance(max_returns, int) and max_returns >= 0:
                next_max = max_returns
            next_return_to = orchestration.return_to
            if isinstance(return_to, str) and return_to.strip():
                next_return_to = return_to.strip()
            next_return_from = orchestration.return_from
            if isinstance(return_from, list) and all(isinstance(x, str) and x.strip() for x in return_from):
                next_return_from = tuple(x.strip() for x in return_from)
            out_orch = OrchestrationConfig(
                preset="linear_v1",
                max_returns=next_max,
                return_to=next_return_to,
                return_from=next_return_from,
            )
        except Exception:
            out_orch = orchestration

    return (
        WizardAnalyzeResult(
            brief_md=brief_md.rstrip() + "\n",
            git=out_git,
            delivery=out_delivery,
            orchestration=out_orch,
        ),
        res.metadata,
    )


@dataclass(frozen=True)
class WizardPacketResult:
    docs: dict[str, str]


def run_setup_wizard_write_packet(
    provider: Provider,
    *,
    goal: str,
    task: TaskConfig | None,
    agent: AgentSpec,
    brief_md: str,
    base_docs: dict[str, str],
    artifacts_dir: Path,
    timeout_s: float,
    idle_timeout_s: float,
    on_event: Callable[[JsonDict], None] | None = None,
) -> tuple[WizardPacketResult | None, JsonDict]:
    """
    Setup wizard (AI): rewrite/expand packet documents for a single agent.
    """
    specialization = agent.specialization.strip() or "(none)"
    task_kind = task.kind if task is not None else "other"
    task_details = task.details_md.strip() if task is not None else ""
    task_details_block = f"Task details:\n{task_details}\n\n" if task_details else ""

    prompt = "".join(
        [
            "You are writing packet documents for an isolated agent.\n\n",
            "Hard constraints:\n",
            "- Keep top-level headings exactly: # ROLE, # TARGET, # RULES, # CONTEXT, # INPUTS.\n",
            "- Do not mention orchestration, other agents, or pipeline order.\n",
            "- Use ASCII only.\n",
            "- Do not output markdown fences.\n",
            "- Keep rules language-agnostic.\n",
            "- Do not suggest destructive workspace commands (no rm -rf, no installs).\n\n",
            "Quality bar:\n",
            "- Make this substantially more detailed than the base templates.\n",
            "- Make the step executable: include a concrete investigation/implementation/validation approach.\n\n",
            "Return exactly one JSON object with keys:\n",
            "- role_md\n",
            "- target_md\n",
            "- rules_md\n",
            "- context_md\n",
            "- inputs_md\n",
            "No extra keys. Values must be non-empty strings.\n\n",
            f"Goal:\n{goal}\n\n",
            f"Task type: {task_kind}\n\n",
            task_details_block,
            f"Agent id: {agent.actor_id}\n",
            f"Template: {agent.template_id}\n",
            f"Specialization: {specialization}\n\n",
            "Setup brief (high-signal decomposition; do not mention orchestration):\n",
            f"{brief_md.strip()}\n\n",
            "Base ROLE.md:\n",
            base_docs["ROLE"].strip() + "\n\n",
            "Base TARGET.md:\n",
            base_docs["TARGET"].strip() + "\n\n",
            "Base RULES.md:\n",
            base_docs["RULES"].strip() + "\n\n",
            "Base CONTEXT.md:\n",
            base_docs["CONTEXT"].strip() + "\n\n",
            "Base INPUTS.md:\n",
            base_docs["INPUTS"].strip() + "\n",
        ]
    )

    res = provider.run(
        prompt,
        artifacts_dir=artifacts_dir,
        timeout_s=timeout_s,
        idle_timeout_s=idle_timeout_s,
        on_event=on_event,
    )

    obj = _extract_first_json_object(res.final_text)
    expected = {"role_md", "target_md", "rules_md", "context_md", "inputs_md"}
    if not isinstance(obj, dict) or set(obj.keys()) != expected:
        return None, res.metadata
    if not all(isinstance(obj[k], str) and obj[k].strip() for k in expected):
        return None, res.metadata

    docs = {
        "ROLE": obj["role_md"].rstrip() + "\n",
        "TARGET": obj["target_md"].rstrip() + "\n",
        "RULES": obj["rules_md"].rstrip() + "\n",
        "CONTEXT": obj["context_md"].rstrip() + "\n",
        "INPUTS": obj["inputs_md"].rstrip() + "\n",
    }
    return WizardPacketResult(docs=docs), res.metadata

