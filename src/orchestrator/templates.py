from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class RoleTemplate:
    template_id: str
    display_name: str
    base_role_md: str
    base_rules_md: str
    base_context_md: str
    build_target_md: Callable[[str, str, str | None], str]


def _target_coder(goal: str, actor_id: str, upstream_actor_id: str | None) -> str:
    _ = upstream_actor_id
    return (
        "# TARGET\n\n"
        f"Goal: {goal}\n\n"
        "Your job:\n"
        "- Propose and implement the minimal set of changes to achieve the goal.\n"
        "- Call out assumptions and risks.\n"
        "- If you need more information, set status=NEEDS_INPUT.\n"
    )


def _target_reviewer(goal: str, actor_id: str, upstream_actor_id: str | None) -> str:
    upstream = upstream_actor_id or "the previous agent"
    return (
        "# TARGET\n\n"
        f"Goal: {goal}\n\n"
        "Your job:\n"
        f"- Review the upstream output from {upstream} (see INPUTS.md).\n"
        "- Identify correctness issues, missing edge cases, and risky assumptions.\n"
        "- Suggest concrete improvements.\n"
        "- If INPUTS.md is insufficient to review, set status=NEEDS_INPUT.\n"
    )


def _target_tester(goal: str, actor_id: str, upstream_actor_id: str | None) -> str:
    upstream = upstream_actor_id or "the previous agent"
    return (
        "# TARGET\n\n"
        f"Goal: {goal}\n\n"
        "Your job:\n"
        f"- Derive a test plan based on the upstream output from {upstream} (see INPUTS.md).\n"
        "- Cover functional tests, negative tests, and key edge cases.\n"
        "- If INPUTS.md lacks implementation details, set status=NEEDS_INPUT.\n"
    )


_CODER = RoleTemplate(
    template_id="coder",
    display_name="Coder (Implementer)",
    base_role_md=(
        "# ROLE\n\n"
        "You are a senior software engineer.\n"
        "You implement changes that achieve the goal in TARGET.\n"
    ),
    base_rules_md=(
        "# RULES\n\n"
        "- Follow ROLE and TARGET.\n"
        "- Use only the information provided in CONTEXT and INPUTS.\n"
        "- Keep the response concise and deterministic.\n"
        "- Output must match REPORT_FORMAT exactly.\n"
    ),
    base_context_md=(
        "# CONTEXT\n\n"
        "You are part of a sequential multi-agent pipeline.\n"
        "Upstream outputs (if any) are provided in INPUTS.md.\n"
    ),
    build_target_md=_target_coder,
)

_REVIEWER = RoleTemplate(
    template_id="reviewer",
    display_name="Reviewer",
    base_role_md=(
        "# ROLE\n\n"
        "You are a senior reviewer.\n"
        "You evaluate upstream output and propose improvements.\n"
    ),
    base_rules_md=(
        "# RULES\n\n"
        "- Follow ROLE and TARGET.\n"
        "- Use only the information provided in CONTEXT and INPUTS.\n"
        "- Be strict and specific; cite concrete issues.\n"
        "- Output must match REPORT_FORMAT exactly.\n"
    ),
    base_context_md=(
        "# CONTEXT\n\n"
        "You are part of a sequential multi-agent pipeline.\n"
        "Your primary input is the upstream agent output in INPUTS.md.\n"
    ),
    build_target_md=_target_reviewer,
)

_TESTER = RoleTemplate(
    template_id="tester",
    display_name="Tester",
    base_role_md=(
        "# ROLE\n\n"
        "You are a senior QA/test engineer.\n"
        "You derive a test plan and validation strategy from upstream output.\n"
    ),
    base_rules_md=(
        "# RULES\n\n"
        "- Follow ROLE and TARGET.\n"
        "- Use only the information provided in CONTEXT and INPUTS.\n"
        "- Prefer actionable test cases.\n"
        "- Output must match REPORT_FORMAT exactly.\n"
    ),
    base_context_md=(
        "# CONTEXT\n\n"
        "You are part of a sequential multi-agent pipeline.\n"
        "Upstream outputs (if any) are provided in INPUTS.md.\n"
    ),
    build_target_md=_target_tester,
)


TEMPLATES: dict[str, RoleTemplate] = {
    _CODER.template_id: _CODER,
    _REVIEWER.template_id: _REVIEWER,
    _TESTER.template_id: _TESTER,
}


def list_template_ids() -> list[str]:
    return sorted(TEMPLATES.keys())


def get_template(template_id: str) -> RoleTemplate:
    return TEMPLATES[template_id]
