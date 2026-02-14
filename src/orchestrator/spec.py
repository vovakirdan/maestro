from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentSpec:
    """
    Minimal agent descriptor used during setup and wizarding.

    - actor_id: stable identifier used in pipeline execution.
    - template_id: one of built-in templates (coder/reviewer/tester/devops) or "custom"/"other".
    - specialization: optional refinement string for built-in templates.
    - custom_role: required when template_id in {"custom", "other"}.
    """

    actor_id: str
    template_id: str
    specialization: str = ""
    custom_role: str = ""
