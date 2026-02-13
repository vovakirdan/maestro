from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from src.core.types import Status
from src.orchestrator.engine import OrchestratorEngine, PlanRequest
from src.orchestrator.setup import run_interactive_setup


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="orchestrator", description="Minimal multi-agent orchestrator (stdlib only)."
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    setup_p = sub.add_parser("setup", help="Interactive quick setup (creates packets + pipeline).")
    run_p = sub.add_parser("run", help="Run the configured pipeline sequentially.")
    for sp in (setup_p, run_p):
        sp.add_argument(
            "--workspace",
            default="workspace",
            help="Workspace directory containing ./orchestrator (default: ./workspace).",
        )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parse_args(sys.argv[1:] if argv is None else argv)
        workspace_dir = Path(args.workspace).expanduser()

        if args.cmd == "setup":
            run_interactive_setup(workspace_dir=workspace_dir)
            return 0

        if args.cmd == "run":

            class _BaseProgressUI:
                def log_line(self, msg: str) -> None:  # pragma: no cover
                    raise NotImplementedError

                def on_event(self, wrapped: dict) -> None:  # pragma: no cover
                    raise NotImplementedError

                def finish(self) -> None:  # pragma: no cover
                    pass

            class SpinnerProgressUI(_BaseProgressUI):
                def __init__(self, *, total_steps: int | None) -> None:
                    self._total_steps = total_steps
                    self._started = time.monotonic()
                    self._frames = ["|", "/", "-", "\\"]
                    self._frame_i = 0
                    self._status_len = 0
                    self._last_summary = ""
                    self._last_line_t = 0.0
                    self._step = ""
                    self._actor_id = ""
                    self._attempt: int | None = None

                def _clear_status(self) -> None:
                    if self._status_len <= 0:
                        return
                    sys.stderr.write("\r" + (" " * self._status_len) + "\r")
                    sys.stderr.flush()
                    self._status_len = 0

                def log_line(self, msg: str) -> None:
                    self._clear_status()
                    sys.stderr.write(msg.rstrip() + "\n")
                    sys.stderr.flush()

                def _summarize(self, ev: dict) -> str | None:
                    t = ev.get("type")
                    if not isinstance(t, str) or not t:
                        return None
                    tl = t.lower()
                    if "delta" in tl or tl in {"token"}:
                        return None
                    if t == "provider":
                        sub = ev.get("event")
                        if isinstance(sub, str) and sub:
                            return f"provider:{sub}"
                        return "provider"
                    if t == "item.completed":
                        item = ev.get("item")
                        if isinstance(item, dict):
                            it = item.get("type")
                            if isinstance(it, str) and it:
                                return f"item.completed:{it}"
                        return "item.completed"
                    if t == "heartbeat":
                        elapsed_s = ev.get("elapsed_s")
                        idle_s = ev.get("idle_s")
                        if isinstance(elapsed_s, (int, float)) and isinstance(idle_s, (int, float)):
                            return f"heartbeat elapsed={elapsed_s:.1f}s idle={idle_s:.1f}s"
                        if isinstance(idle_s, (int, float)):
                            return f"heartbeat idle={idle_s:.1f}s"
                        return "heartbeat"
                    return t

                def on_event(self, wrapped: dict) -> None:
                    if wrapped.get("type") != "provider_event":
                        return
                    step = wrapped.get("step")
                    actor_id = wrapped.get("actor_id")
                    attempt = wrapped.get("attempt")
                    if isinstance(step, str):
                        self._step = step
                    if isinstance(actor_id, str):
                        self._actor_id = actor_id
                    if isinstance(attempt, int):
                        self._attempt = attempt

                    ev = wrapped.get("event")
                    if not isinstance(ev, dict):
                        return
                    summary = self._summarize(ev)
                    if summary is not None:
                        self._last_summary = summary
                        now = time.monotonic()
                        if summary.startswith("heartbeat"):
                            if (now - self._last_line_t) >= 5.0:
                                self._last_line_t = now
                                self.log_line(
                                    f"progress: {self._step} {self._actor_id} attempt {self._attempt or '?'} | {summary}"
                                )
                        elif summary.startswith("item.completed") or summary.endswith("completed"):
                            self.log_line(
                                f"progress: {self._step} {self._actor_id} attempt {self._attempt or '?'} | {summary}"
                            )
                    self.render()

                def render(self) -> None:
                    self._frame_i = (self._frame_i + 1) % len(self._frames)
                    frame = self._frames[self._frame_i]
                    elapsed_s = time.monotonic() - self._started
                    attempt_s = str(self._attempt) if self._attempt is not None else "?"

                    step_idx = "?"
                    if self._step:
                        prefix = self._step.split("_", 1)[0]
                        if prefix.isdigit():
                            step_idx = str(int(prefix))

                    if self._total_steps is None:
                        step_part = f"[{step_idx}]"
                    else:
                        step_part = f"[{step_idx}/{self._total_steps}]"

                    line = (
                        f"[{frame}] {step_part} {self._actor_id} "
                        f"attempt {attempt_s} | {elapsed_s:.1f}s | last={self._last_summary}"
                    )
                    pad = ""
                    if len(line) < self._status_len:
                        pad = " " * (self._status_len - len(line))
                    sys.stderr.write("\r" + line + pad)
                    sys.stderr.flush()
                    self._status_len = len(line)

                def finish(self) -> None:
                    self._clear_status()

            class LineProgressUI(_BaseProgressUI):
                def __init__(self, *, total_steps: int | None) -> None:
                    _ = total_steps
                    self._last_summary = ""
                    self._last_heartbeat = 0.0
                    self._step = ""
                    self._actor_id = ""
                    self._attempt: int | None = None

                def log_line(self, msg: str) -> None:
                    sys.stderr.write(msg.rstrip() + "\n")
                    sys.stderr.flush()

                def _summarize(self, ev: dict) -> str | None:
                    t = ev.get("type")
                    if not isinstance(t, str) or not t:
                        return None
                    tl = t.lower()
                    if "delta" in tl or tl in {"token"}:
                        return None
                    if t == "provider":
                        sub = ev.get("event")
                        if isinstance(sub, str) and sub:
                            return f"provider:{sub}"
                        return "provider"
                    if t == "item.completed":
                        item = ev.get("item")
                        if isinstance(item, dict):
                            it = item.get("type")
                            if isinstance(it, str) and it:
                                return f"item.completed:{it}"
                        return "item.completed"
                    if t == "heartbeat":
                        idle_s = ev.get("idle_s")
                        elapsed_s = ev.get("elapsed_s")
                        if isinstance(idle_s, (int, float)) and isinstance(elapsed_s, (int, float)):
                            return f"heartbeat elapsed={elapsed_s:.1f}s idle={idle_s:.1f}s"
                        return "heartbeat"
                    return t

                def on_event(self, wrapped: dict) -> None:
                    if wrapped.get("type") != "provider_event":
                        return
                    step = wrapped.get("step")
                    actor_id = wrapped.get("actor_id")
                    attempt = wrapped.get("attempt")
                    if isinstance(step, str):
                        self._step = step
                    if isinstance(actor_id, str):
                        self._actor_id = actor_id
                    if isinstance(attempt, int):
                        self._attempt = attempt

                    ev = wrapped.get("event")
                    if not isinstance(ev, dict):
                        return
                    summary = self._summarize(ev)
                    if summary is None:
                        return

                    now = time.monotonic()
                    if summary.startswith("heartbeat"):
                        # Avoid spamming in non-interactive logs.
                        if (now - self._last_heartbeat) < 5.0:
                            return
                        self._last_heartbeat = now

                    if summary == self._last_summary and not summary.startswith("heartbeat"):
                        return
                    self._last_summary = summary

                    attempt_s = str(self._attempt) if self._attempt is not None else "?"
                    sys.stderr.write(
                        f"progress: {self._step} {self._actor_id} attempt {attempt_s} | {summary}\n"
                    )
                    sys.stderr.flush()

            def _prompt_choice(label: str, *, choices: list[str], default: str) -> str:
                choice_map = {c.lower(): c for c in choices}
                while True:
                    raw = input(f"{label} [{default}]: ").strip().lower()
                    if not raw:
                        return default
                    if raw in choice_map:
                        return choice_map[raw]
                    print(f"Please choose one of: {', '.join(choices)}")

            def _read_multiline(*, label: str) -> str:
                print(f"{label} End input with a single '.' line:")
                lines: list[str] = []
                while True:
                    try:
                        line = input()
                    except EOFError:
                        break
                    if line.strip() == ".":
                        break
                    lines.append(line)
                return "\n".join(lines).strip()

            engine = OrchestratorEngine.load(workspace_dir=workspace_dir)
            total_steps: int | None
            if engine.pipeline.orchestration is None:
                total_steps = len(engine.pipeline.actors)
            else:
                total_steps = None

            ui: _BaseProgressUI
            if sys.stderr.isatty():
                ui = SpinnerProgressUI(total_steps=total_steps)
            else:
                ui = LineProgressUI(total_steps=total_steps)

            plan_req: PlanRequest | None = None
            if sys.stdin.isatty():
                allow_auto = engine.pipeline.provider.type != "deterministic"
                plan_choices = ["none", "user"]
                if allow_auto:
                    plan_choices = ["auto", "user", "none"]
                default_plan = "auto" if allow_auto else "none"
                plan_mode = _prompt_choice(
                    f"Plan mode ({'/'.join(plan_choices)})", choices=plan_choices, default=default_plan
                )
                if plan_mode == "user":
                    plan_path = input("Plan file path (optional, press Enter to paste): ").strip()
                    if plan_path:
                        plan_text = Path(plan_path).expanduser().read_text(encoding="utf-8")
                    else:
                        plan_text = _read_multiline(label="Paste plan.")
                    plan_req = PlanRequest(mode="user", text=plan_text)
                elif plan_mode == "auto":
                    plan_req = PlanRequest(mode="auto", text="")
                else:
                    plan_req = PlanRequest(mode="none", text="")

            outcome = engine.run(progress=ui.log_line, on_event=ui.on_event, plan=plan_req)
            ui.finish()
            print("")
            print(f"Run:    {outcome.run_id}")
            print(f"Status: {outcome.status.value}")
            print(f"Dir:    {outcome.run_dir}")
            print("")
            if outcome.final_report is not None:
                print(outcome.final_report.output.rstrip())
                if (
                    outcome.final_report.status == Status.NEEDS_INPUT
                    and outcome.final_report.next_inputs.strip()
                ):
                    print("")
                    print("NEEDS_INPUT:")
                    print(outcome.final_report.next_inputs.rstrip())
            else:
                print(outcome.final_text.rstrip())
            return 0 if outcome.status == Status.OK else 2

        print(f"Unknown command: {args.cmd!r}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
