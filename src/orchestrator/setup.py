from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.core.packet import write_packet_documents
from src.core.runtime import (
    ActorConfig,
    PipelineConfig,
    ProviderConfig,
    orchestrator_root,
    save_pipeline,
)
from src.core.types import JsonDict
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


def _default_report_format() -> str:
    return (
        "FORMAT: ORCH_JSON_V1\n"
        "\n"
        "Return exactly one JSON object (no markdown fences, no extra text) with:\n"
        '- status: "OK" | "NEEDS_INPUT" | "FAILED"\n'
        "- output: string\n"
        "- next_inputs: string (use empty string when not needed)\n"
        "- artifacts: optional list of strings\n"
        "\n"
        "If you cannot proceed, set status=NEEDS_INPUT and describe what you need in next_inputs.\n"
    )


def _template_packet_docs(*, goal: str, role: str) -> dict[str, str]:
    return {
        "ROLE": f"# ROLE\n\n{role}\n",
        "TARGET": f"# TARGET\n\nGoal: {goal}\n",
        "RULES": (
            "# RULES\n\n"
            "- Follow ROLE and TARGET.\n"
            "- Use only the information provided in CONTEXT and INPUTS.\n"
            "- Be concise and deterministic.\n"
            "- Output must match REPORT_FORMAT exactly.\n"
        ),
        "CONTEXT": "# CONTEXT\n\n(Provide any relevant background here.)\n",
        "REPORT_FORMAT": _default_report_format(),
        "INPUTS": "# INPUTS\n\n(Orchestrator will populate this.)\n",
        "NOTES": "# NOTES\n\n(append-only; managed by orchestrator)\n",
    }


def _extract_first_json_object(text: str) -> JsonDict | None:
    import json as _json

    decoder = _json.JSONDecoder()
    idx = text.find("{")
    while idx != -1:
        try:
            obj, _end = decoder.raw_decode(text[idx:])
        except _json.JSONDecodeError:
            idx = text.find("{", idx + 1)
            continue
        return obj if isinstance(obj, dict) else None
    return None


def _generate_packet_docs_via_provider(
    provider: Provider,
    *,
    artifacts_dir: Path,
    goal: str,
    actor_id: str,
    role: str,
    timeout_s: float,
    idle_timeout_s: float,
) -> dict[str, str] | None:
    prompt = (
        "You are generating packet documents for a deterministic multi-agent orchestrator.\n\n"
        f"Goal of orchestration:\n{goal}\n\n"
        f"Agent id: {actor_id}\n"
        f"Agent role:\n{role}\n\n"
        "Return exactly one JSON object with keys:\n"
        "- role_md\n"
        "- target_md\n"
        "- rules_md\n"
        "- context_md\n"
        "Each value must be Markdown text.\n"
        "No markdown fences. No extra keys.\n"
    )

    res = provider.run(
        prompt,
        artifacts_dir=artifacts_dir,
        timeout_s=timeout_s,
        idle_timeout_s=idle_timeout_s,
    )

    obj = _extract_first_json_object(res.final_text)
    if not isinstance(obj, dict):
        return None

    expected = {"role_md", "target_md", "rules_md", "context_md"}
    if set(obj.keys()) != expected:
        return None
    if not all(isinstance(obj[k], str) for k in expected):
        return None

    return {
        "ROLE": obj["role_md"].rstrip() + "\n",
        "TARGET": obj["target_md"].rstrip() + "\n",
        "RULES": obj["rules_md"].rstrip() + "\n",
        "CONTEXT": obj["context_md"].rstrip() + "\n",
    }


@dataclass(frozen=True)
class SetupResult:
    workspace_dir: Path
    orchestrator_dir: Path
    pipeline_path: Path


def run_interactive_setup(*, workspace_dir: Path) -> SetupResult:
    workspace_dir = (
        Path(_prompt_line("Workspace path?", default=str(workspace_dir))).expanduser().resolve()
    )
    orch_dir = orchestrator_root(workspace_dir)
    packets_dir = orch_dir / "packets"
    runs_dir = orch_dir / "runs"

    print("Quick setup will create:")
    print(f"- {packets_dir}")
    print(f"- {runs_dir}")
    print("")

    goal = _prompt_line("What is the goal of orchestration?")
    num_agents = _prompt_int(
        "How many agents do you want? (2-4)", min_value=2, max_value=4, default=2
    )

    roles: list[str] = []
    for i in range(num_agents):
        roles.append(_prompt_line(f"Role for agent_{i+1}"))

    use_codex = _prompt_yes_no("Do you want Codex CLI as provider?", default=False)
    if use_codex:
        cmd_raw = _prompt_line("Codex command (space-separated)", default="codex")
        command = tuple(x for x in cmd_raw.split(" ") if x)
        provider_cfg = ProviderConfig(type="codex_cli", command=command)
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

    auto_generate = False
    if provider_cfg.type == "codex_cli":
        auto_generate = _prompt_yes_no(
            "Auto-generate ROLE/TARGET/RULES/CONTEXT via provider?", default=False
        )

    # Create base directories deterministically.
    runs_dir.mkdir(parents=True, exist_ok=True)
    packets_dir.mkdir(parents=True, exist_ok=True)

    # Provider instance for optional packet generation.
    provider: Provider | None
    if provider_cfg.type == "codex_cli":
        assert provider_cfg.command is not None
        provider = CodexCLIProvider(command=provider_cfg.command)
    else:
        provider = DeterministicProvider()

    actor_cfgs: list[ActorConfig] = []
    for i, role in enumerate(roles, start=1):
        actor_id = f"agent_{i}"
        packet_subdir = actor_id
        actor_cfgs.append(
            ActorConfig(actor_id=actor_id, packet_dir=packet_subdir, include_paths_in_prompt=True)
        )

        packet_dir = packets_dir / packet_subdir
        docs = _template_packet_docs(goal=goal, role=role)

        if auto_generate and provider is not None:
            gen_dir = orch_dir / "setup_artifacts" / actor_id
            gen_dir.mkdir(parents=True, exist_ok=True)
            generated = _generate_packet_docs_via_provider(
                provider,
                artifacts_dir=gen_dir,
                goal=goal,
                actor_id=actor_id,
                role=role,
                timeout_s=provider_cfg.timeout_s,
                idle_timeout_s=provider_cfg.idle_timeout_s,
            )
            if generated is not None:
                docs.update(generated)

        write_packet_documents(packet_dir, docs=docs)

    pipeline = PipelineConfig(version=1, provider=provider_cfg, actors=tuple(actor_cfgs))
    pipeline_path = orch_dir / "pipeline.json"
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
