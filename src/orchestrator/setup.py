from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from src.core.packet import write_packet_documents
from src.core.runtime import (
    ActorConfig,
    OrchestrationConfig,
    PipelineConfig,
    ProviderConfig,
    orchestrator_root,
    save_pipeline,
)
from src.core.types import JsonDict
from src.orchestrator.templates import get_template, list_template_ids
from src.providers.base import Provider
from src.providers.codex_cli import CodexCLIProvider
from src.providers.deterministic import DeterministicProvider


def _prompt_line(label: str, *, default: str | None = None) -> str:
    if default is None:
        suffix = ""
    else:
        suffix = f" [{default}]"
    while True:
        v = input(f"{label}{suffix}: ").strip()
        if v:
            return v
        if default is not None:
            return default


def _prompt_optional_line(label: str) -> str:
    return input(f"{label}: ").strip()


def _prompt_int(label: str, *, min_value: int, max_value: int, default: int) -> int:
    while True:
        raw = _prompt_line(label, default=str(default))
        try:
            n = int(raw)
        except ValueError:
            print(f"Please enter an integer between {min_value} and {max_value}.")
            continue
        if min_value <= n <= max_value:
            return n
        print(f"Please enter an integer between {min_value} and {max_value}.")


def _prompt_yes_no(label: str, *, default: bool) -> bool:
    d = "y" if default else "n"
    while True:
        raw = _prompt_line(label + " (y/n)", default=d).lower()
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("Please enter y or n.")


def _prompt_choice(label: str, *, choices: list[str], default: str) -> str:
    choice_map = {c.lower(): c for c in choices}
    while True:
        raw = _prompt_line(label, default=default).strip().lower()
        if raw in choice_map:
            return choice_map[raw]
        print(f"Please choose one of: {', '.join(choices)}")


def _default_report_format() -> str:
    return (
        "FORMAT: ORCH_JSON_V1\n"
        "\n"
        "Return exactly one JSON object (no markdown fences, no extra text) with:\n"
        '- status: "OK" | "NEEDS_INPUT" | "FAILED"\n'
        "- output: string (your report)\n"
        "- next_inputs: string (use empty string when not needed)\n"
        "- artifacts: optional list of strings (e.g. file paths you changed/created)\n"
        "\n"
        "Status semantics:\n"
        "- OK: you completed your step.\n"
        "- FAILED: changes are required (use next_inputs for actionable change requests / repro steps).\n"
        "- NEEDS_INPUT: you are blocked; ask precise questions in next_inputs.\n"
        "\n"
        "If you cannot proceed, set status=NEEDS_INPUT and describe what you need in next_inputs.\n"
    )


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


@dataclass(frozen=True)
class AgentSpec:
    actor_id: str
    template_id: str  # coder/reviewer/tester/custom
    specialization: str
    custom_role: str


def _base_packet_docs(
    *,
    workspace_dir: Path,
    goal: str,
    agent: AgentSpec,
    is_first: bool,
) -> dict[str, str]:
    if agent.template_id == "custom":
        role_md = "# ROLE\n\n" + agent.custom_role.strip() + "\n"
        rules_md = "# RULES\n\nFollow ROLE, TARGET, and REPORT_FORMAT.\n"
        context_md = "# CONTEXT\n\nYou have access to a workspace/repository to modify.\n"
        target_md = "# TARGET\n\n" f"Goal:\n{goal}\n\nDescribe what you will do in this step.\n"
    else:
        tmpl = get_template(agent.template_id)
        role_md = tmpl.base_role_md
        if agent.specialization.strip():
            role_md = (
                role_md.rstrip() + "\n\n" + f"Specialization: {agent.specialization.strip()}\n"
            )
        rules_md = tmpl.base_rules_md
        context_md = tmpl.base_context_md
        target_md = tmpl.build_target_md(goal, agent.actor_id, None)

    context_md = (
        context_md.rstrip()
        + "\n\n"
        + f"Goal: {goal}\n"
        + f"Workspace root: {workspace_dir.as_posix()}\n"
        + "\n"
    )

    if is_first:
        inputs_md = (
            "# INPUTS\n\n"
            f"Initial request:\n{goal}\n\n"
            "Additional constraints:\n"
            "- (none provided)\n\n"
            "Notes:\n"
            "- Use this file for any extra requirements or clarifications.\n"
        )
    else:
        inputs_md = (
            "# INPUTS\n\n"
            "This file will contain any material you should act on for this step.\n"
            "If it is empty/insufficient, set status=NEEDS_INPUT.\n"
        )

    return {
        "ROLE": role_md.rstrip() + "\n",
        "TARGET": target_md.rstrip() + "\n",
        "RULES": rules_md.rstrip() + "\n",
        "CONTEXT": context_md.rstrip() + "\n",
        "REPORT_FORMAT": _default_report_format(),
        "INPUTS": inputs_md,
        "NOTES": "# NOTES\n\n(append-only; managed by orchestrator)\n",
    }


def _refine_packet_docs_via_provider(
    provider: Provider,
    *,
    artifacts_dir: Path,
    goal: str,
    agent: AgentSpec,
    base_role_md: str,
    base_target_md: str,
    base_rules_md: str,
    base_context_md: str,
    timeout_s: float,
    idle_timeout_s: float,
) -> tuple[dict[str, str] | None, JsonDict]:
    specialization = agent.specialization.strip() or "(none)"
    prompt = (
        "You are refining packet documents (ROLE/TARGET/RULES/CONTEXT) for an isolated agent.\n\n"
        "Goal: produce high-quality instructions with a strong quality bar.\n"
        "The documents should be practical, specific, and deterministic.\n\n"
        "Hard constraints:\n"
        "- Keep the top-level headings exactly: # ROLE, # TARGET, # RULES, # CONTEXT.\n"
        "- Do not mention other agents, pipeline order, or orchestration.\n"
        "- Keep rules language-agnostic (no language/framework-specific advice).\n"
        "- Do not output markdown fences.\n"
        "- Do not change the output protocol: the agent must follow REPORT_FORMAT.md.\n\n"
        "Content guidelines (make this substantially more detailed than the base docs):\n"
        "- ROLE: responsibilities, non-goals, decision making, definition of done.\n"
        "- TARGET: explicit deliverables and success criteria for this step.\n"
        "- RULES: correctness, minimal change, avoiding hallucinations, validation expectations.\n"
        "- CONTEXT: how to use INPUTS.md, how to handle ambiguity, where to look in workspace.\n\n"
        "Return exactly one JSON object with keys:\n"
        "- role_md\n"
        "- target_md\n"
        "- rules_md\n"
        "- context_md\n"
        "No extra keys. Values must be non-empty strings.\n\n"
        f"Goal:\n{goal}\n\n"
        f"Agent id: {agent.actor_id}\n"
        f"Template: {agent.template_id}\n"
        f"Specialization: {specialization}\n\n"
        "REPORT_FORMAT.md (do not edit, but your docs must be consistent with it):\n"
        f"{_default_report_format()}\n"
        "\n"
        "Current ROLE.md:\n"
        f"{base_role_md}\n\n"
        "Current TARGET.md:\n"
        f"{base_target_md}\n\n"
        "Current RULES.md:\n"
        f"{base_rules_md}\n\n"
        "Current CONTEXT.md:\n"
        f"{base_context_md}\n"
    )

    res = provider.run(
        prompt,
        artifacts_dir=artifacts_dir,
        timeout_s=timeout_s,
        idle_timeout_s=idle_timeout_s,
    )

    obj = _extract_first_json_object(res.final_text)
    expected = {"role_md", "target_md", "rules_md", "context_md"}
    if not isinstance(obj, dict) or set(obj.keys()) != expected:
        return None, res.metadata
    if not all(isinstance(obj[k], str) and obj[k].strip() for k in expected):
        return None, res.metadata

    return (
        {
            "ROLE": obj["role_md"].rstrip() + "\n",
            "TARGET": obj["target_md"].rstrip() + "\n",
            "RULES": obj["rules_md"].rstrip() + "\n",
            "CONTEXT": obj["context_md"].rstrip() + "\n",
        },
        res.metadata,
    )


@dataclass(frozen=True)
class SetupResult:
    workspace_dir: Path
    orchestrator_dir: Path
    pipeline_path: Path


def run_interactive_setup(*, workspace_dir: Path) -> SetupResult:
    workspace_dir = Path(_prompt_line("Workspace path?", default=str(workspace_dir))).expanduser()
    workspace_dir = workspace_dir.resolve()
    orch_dir = orchestrator_root(workspace_dir)
    packets_dir = orch_dir / "packets"
    runs_dir = orch_dir / "runs"
    pipeline_path = orch_dir / "pipeline.json"

    if orch_dir.exists() and (pipeline_path.exists() or packets_dir.exists()):
        overwrite = _prompt_yes_no(
            "Overwrite existing orchestrator config?",
            default=False,
        )
        if not overwrite:
            print("Setup aborted (no changes made).")
            return SetupResult(
                workspace_dir=workspace_dir,
                orchestrator_dir=orch_dir,
                pipeline_path=pipeline_path,
            )
        if packets_dir.exists():
            shutil.rmtree(packets_dir)

    print("Quick setup will create:")
    print(f"- {packets_dir}")
    print(f"- {runs_dir}")
    print("")

    goal = _prompt_line("What is the goal of orchestration?")
    preset = _prompt_choice(
        "Orchestration preset (crt/custom)",
        choices=["crt", "custom"],
        default="crt",
    )
    max_returns = 3
    if preset == "crt":
        max_returns = _prompt_int(
            "Max returns (review/test -> coder)",
            min_value=0,
            max_value=10,
            default=3,
        )
        num_agents = 3
    else:
        num_agents = _prompt_int(
            "How many agents do you want? (2-4)", min_value=2, max_value=4, default=2
        )

    use_codex = _prompt_yes_no("Do you want Codex CLI as provider?", default=False)
    if use_codex:
        cmd_raw = _prompt_line(
            "Codex command (space-separated)",
            default="codex exec --json --full-auto",
        )
        command = tuple(cmd_raw.split())
        if "--json" not in command:
            print("WARN: Codex command does not include '--json'; JSONL parsing may fail.")
        provider_cfg = ProviderConfig(
            type="codex_cli",
            command=command,
            timeout_s=600.0,
            idle_timeout_s=300.0,
        )
    else:
        provider_cfg = ProviderConfig(type="deterministic")

    timeout_s = float(
        _prompt_line("Hard timeout seconds", default=str(int(provider_cfg.timeout_s)))
    )
    idle_timeout_s = float(
        _prompt_line("Idle timeout seconds", default=str(int(provider_cfg.idle_timeout_s)))
    )
    provider_cfg = ProviderConfig(
        type=provider_cfg.type,
        command=provider_cfg.command,
        timeout_s=timeout_s,
        idle_timeout_s=idle_timeout_s,
    )

    refine_now = False
    if provider_cfg.type == "codex_cli":
        refine_now = _prompt_yes_no("Use provider to refine packet documents now?", default=True)

    # Choose templates and specializations.
    agents: list[AgentSpec] = []
    if preset == "crt":
        for template_id in ("coder", "reviewer", "tester"):
            actor_id = template_id
            specialization = _prompt_optional_line(f"Specialization for {actor_id} (optional)")
            agents.append(
                AgentSpec(
                    actor_id=actor_id,
                    template_id=template_id,
                    specialization=specialization,
                    custom_role="",
                )
            )
    else:
        template_ids = list_template_ids() + ["custom"]
        for i in range(1, num_agents + 1):
            actor_id = f"agent_{i}"
            default_template = "custom"
            if i == 1:
                default_template = "coder"
            elif i == 2:
                default_template = "reviewer"
            elif i == 3:
                default_template = "tester"

            template_id = _prompt_choice(
                f"Template for {actor_id} ({'/'.join(template_ids)})",
                choices=template_ids,
                default=default_template,
            )
            if template_id == "custom":
                role = _prompt_line(f"Role for {actor_id}")
                agents.append(
                    AgentSpec(
                        actor_id=actor_id,
                        template_id="custom",
                        specialization="",
                        custom_role=role,
                    )
                )
            else:
                specialization = _prompt_optional_line(f"Specialization for {actor_id} (optional)")
                agents.append(
                    AgentSpec(
                        actor_id=actor_id,
                        template_id=template_id,
                        specialization=specialization,
                        custom_role="",
                    )
                )

    # Create base directories deterministically.
    runs_dir.mkdir(parents=True, exist_ok=True)
    packets_dir.mkdir(parents=True, exist_ok=True)

    # Provider instance for optional refinement.
    provider: Provider
    if provider_cfg.type == "codex_cli":
        assert provider_cfg.command is not None
        provider = CodexCLIProvider(command=provider_cfg.command, cwd=workspace_dir)
    else:
        provider = DeterministicProvider()

    actor_cfgs: list[ActorConfig] = []
    for i, agent in enumerate(agents, start=1):
        actor_cfgs.append(
            ActorConfig(
                actor_id=agent.actor_id,
                packet_dir=agent.actor_id,
                include_paths_in_prompt=True,
            )
        )

        docs = _base_packet_docs(
            workspace_dir=workspace_dir,
            goal=goal,
            agent=agent,
            is_first=i == 1,
        )

        if refine_now and provider_cfg.type == "codex_cli":
            gen_dir = orch_dir / "setup_artifacts" / agent.actor_id
            gen_dir.mkdir(parents=True, exist_ok=True)
            refined, meta = _refine_packet_docs_via_provider(
                provider,
                artifacts_dir=gen_dir,
                goal=goal,
                agent=agent,
                base_role_md=docs["ROLE"],
                base_target_md=docs["TARGET"],
                base_rules_md=docs["RULES"],
                base_context_md=docs["CONTEXT"],
                timeout_s=provider_cfg.timeout_s,
                idle_timeout_s=provider_cfg.idle_timeout_s,
            )
            if refined is not None:
                docs.update(refined)
            else:
                timed_out = bool(meta.get("timed_out"))
                idle_timed_out = bool(meta.get("idle_timed_out"))
                hint = ""
                if timed_out or idle_timed_out:
                    hint = f" (timed_out={timed_out}, idle_timed_out={idle_timed_out})"
                print(
                    "WARN: provider refinement failed for "
                    f"{agent.actor_id}; using base templates.{hint}"
                )
                print(f"WARN: see artifacts in {gen_dir}")

        packet_dir = packets_dir / agent.actor_id
        write_packet_documents(packet_dir, docs=docs)

    orchestration = None
    if preset == "crt":
        orchestration = OrchestrationConfig(preset="crt_v1", max_returns=max_returns)
    pipeline = PipelineConfig(
        version=1,
        provider=provider_cfg,
        actors=tuple(actor_cfgs),
        orchestration=orchestration,
    )
    save_pipeline(pipeline_path, pipeline)

    print("")
    print("Setup complete.")
    print(f"Workspace: {workspace_dir}")
    print(f"Pipeline:  {pipeline_path}")

    return SetupResult(
        workspace_dir=workspace_dir,
        orchestrator_dir=orch_dir,
        pipeline_path=pipeline_path,
    )
