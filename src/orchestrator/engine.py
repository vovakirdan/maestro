from __future__ import annotations

import json
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from src.core.actor import Actor
from src.core.packet import Packet
from src.core.runtime import PipelineConfig, generate_run_id, load_pipeline, orchestrator_root
from src.core.types import JsonDict, Status, StructuredReport
from src.orchestrator.validator import ReportValidator, retry_instructions
from src.providers.base import Provider
from src.providers.codex_cli import CodexCLIProvider
from src.providers.deterministic import DeterministicProvider


def _now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _write_json(path: Path, obj: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class TimelineWriter:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: JsonDict) -> None:
        event = dict(event)
        event.setdefault("ts_utc", _now_iso_utc())
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, sort_keys=True) + "\n")


@dataclass(frozen=True)
class RunOutcome:
    run_id: str
    status: Status
    run_dir: Path
    final_report: StructuredReport | None
    final_text: str


@dataclass
class OrchestratorEngine:
    orchestrator_dir: Path
    pipeline: PipelineConfig

    @staticmethod
    def load(*, workspace_dir: Path) -> OrchestratorEngine:
        orch_dir = orchestrator_root(workspace_dir.resolve())
        pipeline_path = orch_dir / "pipeline.json"
        pipeline = load_pipeline(pipeline_path)
        return OrchestratorEngine(orchestrator_dir=orch_dir, pipeline=pipeline)

    def _make_provider(self) -> Provider:
        cfg = self.pipeline.provider
        if cfg.type == "deterministic":
            return DeterministicProvider()
        if cfg.type == "codex_cli":
            assert cfg.command is not None
            return CodexCLIProvider(command=cfg.command)
        raise ValueError(f"Unsupported provider type: {cfg.type!r}")

    def run(self, *, progress: Callable[[str], None] | None = None) -> RunOutcome:
        def log(msg: str) -> None:
            if progress:
                progress(msg)

        provider = self._make_provider()
        validator = ReportValidator()

        run_id = generate_run_id()
        run_dir = self.orchestrator_dir / "runs" / run_id
        steps_dir = run_dir / "steps"
        timeline_path = run_dir / "timeline.jsonl"
        state_path = run_dir / "state.json"
        timeline = TimelineWriter(timeline_path)

        template_packets_dir = self.orchestrator_dir / "packets"
        run_packets_dir = run_dir / "packets"

        steps_dir.mkdir(parents=True, exist_ok=False)
        shutil.copytree(template_packets_dir, run_packets_dir)

        state: JsonDict = {
            "run_id": run_id,
            "started_at_utc": _now_iso_utc(),
            "status": "RUNNING",
            "steps": [],
        }
        _write_json(state_path, state)
        timeline.append({"type": "run_started", "run_id": run_id})

        final_report: StructuredReport | None = None
        final_text: str = ""
        overall_status = Status.OK

        for idx, actor_cfg in enumerate(self.pipeline.actors, start=1):
            actor_id = actor_cfg.actor_id
            step_name = f"{idx:02d}_{actor_id}"
            step_dir = steps_dir / step_name
            step_dir.mkdir(parents=True, exist_ok=False)

            log(f"[{idx}/{len(self.pipeline.actors)}] Running {actor_id} ...")
            timeline.append(
                {"type": "step_started", "run_id": run_id, "step": step_name, "actor_id": actor_id}
            )

            packet_dir = run_packets_dir / actor_cfg.packet_dir
            packet = Packet.load(packet_dir)
            report_format_text = packet.documents["REPORT_FORMAT"].text

            actor = Actor(
                actor_id=actor_id,
                packet=packet,
                provider=provider,
                include_paths_in_prompt=actor_cfg.include_paths_in_prompt,
            )

            attempt_errors: tuple[str, ...] = tuple()
            validated_report: StructuredReport | None = None
            validated_text: str = ""

            for attempt in (1, 2):
                attempt_dir = step_dir / f"attempt_{attempt}"
                extra = None
                if attempt == 2:
                    extra = retry_instructions(attempt_errors)
                res = actor.run(
                    artifacts_dir=attempt_dir,
                    timeout_s=self.pipeline.provider.timeout_s,
                    idle_timeout_s=self.pipeline.provider.idle_timeout_s,
                    extra_instructions=extra,
                )

                validated_text = res.final_text
                v = validator.validate(
                    report_format_text=report_format_text, output_text=validated_text
                )
                _write_json(
                    attempt_dir / "validation.json",
                    {
                        "ok": v.ok,
                        "errors": list(v.errors),
                        "provider_metadata": res.metadata,
                    },
                )
                timeline.append(
                    {
                        "type": "attempt_finished",
                        "run_id": run_id,
                        "step": step_name,
                        "actor_id": actor_id,
                        "attempt": attempt,
                        "validation_ok": v.ok,
                        "provider_metadata": res.metadata,
                    }
                )

                if v.ok and v.report is not None:
                    validated_report = v.report
                    break
                attempt_errors = v.errors

            # Canonical step outputs.
            (step_dir / "final.txt").write_text(validated_text, encoding="utf-8")
            if validated_report is not None:
                _write_json(
                    step_dir / "report.json",
                    {
                        "status": validated_report.status.value,
                        "output": validated_report.output,
                        "next_inputs": validated_report.next_inputs,
                        "artifacts": list(validated_report.artifacts),
                    },
                )

            # Append to NOTES.md (append-only within the run's packet copy).
            notes_path = packet_dir / "NOTES.md"
            with notes_path.open("a", encoding="utf-8") as f:
                f.write(f"\n## {run_id} {step_name}\n")
                if validated_report is None:
                    f.write("STATUS: FAILED (invalid report format)\n")
                    for e in attempt_errors:
                        f.write(f"- {e}\n")
                else:
                    f.write(f"STATUS: {validated_report.status.value}\n")

            if validated_report is None:
                overall_status = Status.FAILED
                state["status"] = overall_status.value
                state["finished_at_utc"] = _now_iso_utc()
                state["steps"].append(
                    {
                        "step": step_name,
                        "actor_id": actor_id,
                        "status": "FAILED",
                        "reason": "invalid_report_format",
                        "errors": list(attempt_errors),
                    }
                )
                _write_json(state_path, state)
                timeline.append(
                    {
                        "type": "step_failed",
                        "run_id": run_id,
                        "step": step_name,
                        "actor_id": actor_id,
                        "reason": "invalid_report_format",
                        "errors": list(attempt_errors),
                    }
                )
                break

            # Step succeeded with a validated report.
            final_report = validated_report
            final_text = validated_text
            step_status = validated_report.status
            state["steps"].append(
                {
                    "step": step_name,
                    "actor_id": actor_id,
                    "status": step_status.value,
                }
            )
            _write_json(state_path, state)
            timeline.append(
                {"type": "step_finished", "run_id": run_id, "step": step_name, "actor_id": actor_id}
            )

            if step_status == Status.OK:
                # Propagate raw final text to next step's INPUTS.md (run-local packet copy).
                if idx < len(self.pipeline.actors):
                    next_cfg = self.pipeline.actors[idx]
                    next_packet_dir = run_packets_dir / next_cfg.packet_dir
                    next_inputs_path = next_packet_dir / "INPUTS.md"
                    next_inputs_path.write_text(
                        f"# INPUTS\n\nUpstream output from {step_name}:\n\n{validated_text}\n",
                        encoding="utf-8",
                    )
                continue

            overall_status = step_status
            state["status"] = overall_status.value
            state["finished_at_utc"] = _now_iso_utc()
            _write_json(state_path, state)
            timeline.append(
                {
                    "type": "run_stopped",
                    "run_id": run_id,
                    "status": overall_status.value,
                    "step": step_name,
                    "actor_id": actor_id,
                }
            )
            break

        if overall_status == Status.OK:
            state["status"] = overall_status.value
            state["finished_at_utc"] = _now_iso_utc()
            _write_json(state_path, state)
            timeline.append(
                {"type": "run_finished", "run_id": run_id, "status": overall_status.value}
            )

        (run_dir / "final.txt").write_text(final_text, encoding="utf-8")
        if final_report is not None:
            _write_json(
                run_dir / "final_report.json",
                {
                    "status": final_report.status.value,
                    "output": final_report.output,
                    "next_inputs": final_report.next_inputs,
                    "artifacts": list(final_report.artifacts),
                },
            )

        return RunOutcome(
            run_id=run_id,
            status=overall_status,
            run_dir=run_dir,
            final_report=final_report,
            final_text=final_text,
        )
