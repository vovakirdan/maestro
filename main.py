from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from src.core.types import Status
from src.orchestrator.daemon import (
    DEFAULT_POLL_INTERVAL,
    OrchestratorDaemon,
    render_status_lines,
    resolve_daemon_data_dir,
)
from src.orchestrator.engine import GitPolicy, OrchestratorEngine, PlanRequest
from src.orchestrator.setup import run_interactive_setup


def _add_workspace_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--workspace",
        default="workspace",
        help="Workspace directory containing ./orchestrator (default: ./workspace).",
    )


def _add_daemon_data_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--daemon-data-dir",
        default=None,
        help="Daemon base dir (default: ORCHESTRATOR_DAEMON_DIR or ~/.cache/orchestrator).",
    )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="orchestrator", description="Minimal multi-agent orchestrator (stdlib only)."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    setup_p = sub.add_parser("setup", help="Interactive quick setup (creates packets + pipeline).")
    _add_workspace_arg(setup_p)

    ticket_p = sub.add_parser("ticket", help="Manage tickets (prepare task presets without running).")
    ticket_sub = ticket_p.add_subparsers(dest="ticket_cmd", required=True)
    ticket_start_p = ticket_sub.add_parser("start", help="Create a new ticket and prepare its pipeline.")
    _add_workspace_arg(ticket_start_p)

    run_p = sub.add_parser("run", help="Run the configured pipeline sequentially.")
    _add_workspace_arg(run_p)
    _add_daemon_data_arg(run_p)
    run_p.add_argument(
        "--timeout-s",
        type=float,
        default=None,
        help="Override provider hard timeout seconds (0 disables hard timeout).",
    )
    run_p.add_argument(
        "--task",
        default=None,
        help="Task id under <workspace>/orchestrator/tasks/<task_id> (default: CURRENT_TASK or legacy pipeline).",
    )
    run_p.add_argument(
        "--idle-timeout-s",
        type=float,
        default=None,
        help="Override provider idle timeout seconds (0 disables idle timeout).",
    )
    run_p.add_argument(
        "--ask-git",
        action="store_true",
        help="Prompt for git policy even when pipeline.git_defaults is present.",
    )
    run_p.add_argument(
        "--foreground",
        action="store_true",
        help="Run in current process even if daemon is available.",
    )
    run_p.add_argument(
        "--async",
        dest="async_mode",
        action="store_true",
        help="Force async execution through daemon (requires daemon running).",
    )
    run_p.add_argument(
        "--wait",
        action="store_true",
        help="Wait for async run result and print final status.",
    )
    run_p.add_argument(
        "--poll-interval",
        type=float,
        default=DEFAULT_POLL_INTERVAL,
        help="Poll interval for async wait/tail in seconds (default: 1.0).",
    )

    status_p = sub.add_parser("status", help="Show run status from daemon store.")
    _add_workspace_arg(status_p)
    _add_daemon_data_arg(status_p)
    status_p.add_argument("--run-id", default=None, help="Optional run id to show run state.")
    status_p.add_argument("--json", action="store_true", help="Output JSON.")

    history_p = sub.add_parser("history", help="Show daemon run history for a workspace.")
    _add_workspace_arg(history_p)
    _add_daemon_data_arg(history_p)
    history_p.add_argument("--task", default=None, help="Filter by task id.")
    history_p.add_argument(
        "--status",
        default=None,
        choices=["PENDING", "RUNNING", "OK", "FAILED", "NEEDS_INPUT"],
        help="Filter by run status.",
    )
    history_p.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum rows to show.",
    )
    history_p.add_argument("--json", action="store_true", help="Output JSON.")

    daemon_p = sub.add_parser("daemon", help="Manage background orchestrator daemon.")
    daemon_sub = daemon_p.add_subparsers(dest="daemon_cmd", required=True)

    d_start = daemon_sub.add_parser("start", help="Start daemon for the workspace.")
    _add_workspace_arg(d_start)
    _add_daemon_data_arg(d_start)
    d_start.add_argument(
        "--poll-interval",
        type=float,
        default=DEFAULT_POLL_INTERVAL,
        help="Daemon polling interval in seconds (default: 1.0).",
    )

    d_stop = daemon_sub.add_parser("stop", help="Stop daemon for the workspace.")
    _add_workspace_arg(d_stop)
    _add_daemon_data_arg(d_stop)
    d_stop.add_argument("--force", action="store_true", help="Kill forcefully if graceful stop fails.")

    d_status = daemon_sub.add_parser("status", help="Show daemon status for workspace.")
    _add_workspace_arg(d_status)
    _add_daemon_data_arg(d_status)
    d_status.add_argument("--json", action="store_true", help="Output JSON.")

    d_runs = daemon_sub.add_parser("runs", help="List daemon runs for workspace.")
    _add_workspace_arg(d_runs)
    _add_daemon_data_arg(d_runs)
    d_runs.add_argument("--task", default=None, help="Filter by task id.")
    d_runs.add_argument(
        "--status",
        default=None,
        choices=["PENDING", "RUNNING", "OK", "FAILED", "NEEDS_INPUT"],
        help="Filter by run status.",
    )
    d_runs.add_argument("--limit", type=int, default=20, help="Maximum rows.")
    d_runs.add_argument("--json", action="store_true", help="Output JSON.")

    d_tail = daemon_sub.add_parser("tail", help="Print daemon run timeline tail.")
    _add_workspace_arg(d_tail)
    _add_daemon_data_arg(d_tail)
    d_tail.add_argument("--run-id", required=True, help="Run id to show.")
    d_tail.add_argument("--lines", type=int, default=200, help="Initial lines to show.")
    d_tail.add_argument("--follow", action="store_true", help="Follow mode.")
    d_tail.add_argument(
        "--poll-interval",
        type=float,
        default=DEFAULT_POLL_INTERVAL,
        help="Polling interval in seconds (default: 1.0).",
    )

    d_worker = daemon_sub.add_parser(
        "worker",
        help=argparse.SUPPRESS,
    )
    _add_workspace_arg(d_worker)
    _add_daemon_data_arg(d_worker)
    d_worker.add_argument(
        "--poll-interval",
        type=float,
        default=DEFAULT_POLL_INTERVAL,
        help="Daemon polling interval in seconds (default: 1.0).",
    )

    return p.parse_args(argv)


class _BaseProgressUI:
    def log_line(self, msg: str) -> None:  # pragma: no cover
        raise NotImplementedError

    def on_event(self, wrapped: dict) -> None:  # pragma: no cover
        raise NotImplementedError

    def finish(self) -> None:  # pragma: no cover
        pass


class _SpinnerProgressUI(_BaseProgressUI):
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

    @staticmethod
    def _summarize(ev: dict) -> str | None:
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
            f"[{frame}] {step_part} {self._actor_id} attempt {attempt_s} | "
            f"{elapsed_s:.1f}s | last={self._last_summary}"
        )
        pad = ""
        if len(line) < self._status_len:
            pad = " " * (self._status_len - len(line))
        sys.stderr.write("\r" + line + pad)
        sys.stderr.flush()
        self._status_len = len(line)

    def finish(self) -> None:
        self._clear_status()


class _LineProgressUI(_BaseProgressUI):
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

    @staticmethod
    def _summarize(ev: dict) -> str | None:
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


def _wait_for_daemon_run(
    daemon: OrchestratorDaemon,
    run_id: str,
    *,
    poll_interval: float,
    follow: bool = False,
    print_timeline: bool = False,
) -> dict[str, Any]:
    done = {"OK", "FAILED", "NEEDS_INPUT", "CANCELLED"}
    seen = 0
    state: dict[str, Any] = {}
    while True:
        state = daemon.read_run_state(run_id)
        status = state.get("status")
        if print_timeline:
            lines = daemon.read_run_timeline(run_id, limit=0)
            if len(lines) > seen:
                for line in lines[seen:]:
                    print(line)
                seen = len(lines)
        if status in done:
            break
        if not follow:
            break
        time.sleep(max(0.2, poll_interval))
    return state


def _read_run_summary(run_dir: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    final_txt = run_dir / "final.txt"
    if final_txt.exists():
        summary["final_text"] = final_txt.read_text(encoding="utf-8")
    report_path = run_dir / "final_report.json"
    if report_path.exists():
        try:
            summary["final_report"] = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return summary


def _print_final_for_run(run_id: str, run_dir: Path, state: dict[str, Any]) -> int:
    status = str(state.get("status", "UNKNOWN"))
    print(f"Run:    {run_id}")
    print(f"Status: {status}")
    print(f"Dir:    {run_dir}")
    print("")

    summary = _read_run_summary(run_dir)
    report_raw = summary.get("final_report")
    if isinstance(report_raw, dict):
        output = report_raw.get("output", "")
        if isinstance(output, str):
            print(output.rstrip())
        next_inputs = report_raw.get("next_inputs")
        if status == "NEEDS_INPUT" and isinstance(next_inputs, str) and next_inputs.strip():
            print("")
            print("NEEDS_INPUT:")
            print(next_inputs.rstrip())
    else:
        text = summary.get("final_text")
        if isinstance(text, str):
            print(text.rstrip())

    if status == "NEEDS_INPUT":
        ni_path = run_dir / "NEEDS_INPUT.md"
        if ni_path.exists():
            print(f"NEEDS_INPUT file: {ni_path}")
        esc_path = run_dir / "escalation" / "escalation.md"
        if esc_path.exists():
            print(f"Escalation file: {esc_path}")

    return 0 if status == "OK" else 2


def _build_default_daemon(workspace_dir: Path, args: argparse.Namespace) -> OrchestratorDaemon:
    daemon_data_dir = resolve_daemon_data_dir(args.daemon_data_dir)
    poll_interval = max(0.2, float(args.poll_interval)) if getattr(args, "poll_interval", None) else DEFAULT_POLL_INTERVAL
    return OrchestratorDaemon(
        workspace_dir=workspace_dir,
        daemon_data_dir=daemon_data_dir,
        poll_interval_s=poll_interval,
    )


def _records_to_dict(records: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in records:
        if hasattr(r, "__dict__"):
            out.append(dict(r.__dict__))
        elif isinstance(r, dict):
            out.append(r)
    return out


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parse_args(sys.argv[1:] if argv is None else argv)
        workspace_dir = Path(args.workspace).expanduser()

        if args.cmd == "setup":
            run_interactive_setup(workspace_dir=workspace_dir)
            return 0

        if args.cmd == "ticket":
            if args.ticket_cmd == "start":
                run_interactive_setup(workspace_dir=workspace_dir, mode="ticket")
                return 0
            print("Unknown ticket command.", file=sys.stderr)
            return 2

        if args.cmd == "status":
            daemon = _build_default_daemon(workspace_dir, args)
            run_id = args.run_id
            if run_id:
                state = daemon.read_run_state(run_id)
                if not state:
                    print(f"Run not found: {run_id}", file=sys.stderr)
                    return 2
                if args.json:
                    print(json.dumps(state, sort_keys=True, indent=2))
                    return 0
                for line in render_status_lines(state):
                    print(line)
                return 0

            status = daemon.status()
            if args.json:
                print(json.dumps(status, sort_keys=True, indent=2))
                return 0
            for line in render_status_lines(status):
                print(line)
            return 0

        if args.cmd == "history":
            daemon = _build_default_daemon(workspace_dir, args)
            records = daemon.list_runs(task=args.task, status_filter=args.status, limit=args.limit)
            if args.json:
                print(json.dumps(_records_to_dict(records), sort_keys=True, indent=2))
                return 0
            if not records:
                print("No runs in daemon store.")
                return 0
            for rec in records:
                print(f"{rec.run_id} status={rec.status} task={rec.task_id} created={rec.created_at_utc}")
            return 0

        if args.cmd == "daemon":
            if args.daemon_cmd == "start":
                daemon = _build_default_daemon(workspace_dir, args)
                try:
                    pid = daemon.start()
                    print(f"daemon started: pid={pid}")
                    return 0
                except RuntimeError as e:
                    print(f"daemon start failed: {e}", file=sys.stderr)
                    return 2

            if args.daemon_cmd == "stop":
                daemon = _build_default_daemon(workspace_dir, args)
                ok = daemon.stop(force=args.force)
                if ok:
                    print("daemon stopped")
                    return 0
                print("daemon was not running", file=sys.stderr)
                return 2

            if args.daemon_cmd == "status":
                daemon = _build_default_daemon(workspace_dir, args)
                status = daemon.status()
                if args.json:
                    print(json.dumps(status, sort_keys=True, indent=2))
                    return 0
                for line in render_status_lines(status):
                    print(line)
                return 0

            if args.daemon_cmd == "runs":
                daemon = _build_default_daemon(workspace_dir, args)
                records = daemon.list_runs(
                    task=args.task,
                    status_filter=args.status,
                    limit=args.limit,
                )
                if args.json:
                    print(json.dumps(_records_to_dict(records), sort_keys=True, indent=2))
                    return 0
                if not records:
                    print("No runs in daemon store.")
                    return 0
                for rec in records:
                    print(f"{rec.run_id} status={rec.status} task={rec.task_id} created={rec.created_at_utc}")
                return 0

            if args.daemon_cmd == "tail":
                daemon = _build_default_daemon(workspace_dir, args)
                run_id = args.run_id
                lines = daemon.read_run_timeline(run_id, limit=max(1, args.lines))
                for line in lines:
                    print(line)
                if not args.follow:
                    return 0
                _wait_for_daemon_run(
                    daemon,
                    run_id,
                    poll_interval=max(0.2, float(args.poll_interval)),
                    follow=True,
                    print_timeline=True,
                )
                return 0

            if args.daemon_cmd == "worker":
                daemon = _build_default_daemon(workspace_dir, args)
                daemon.run_worker()
                return 0

            print("Unknown daemon command.", file=sys.stderr)
            return 2

        if args.cmd == "run":
            engine = OrchestratorEngine.load(workspace_dir=workspace_dir, task_id=args.task)

            daemon = _build_default_daemon(workspace_dir, args)
            running_in_daemon, _ = daemon.is_running()

            total_steps: int | None = (
                len(engine.pipeline.actors) if engine.pipeline.orchestration is None else None
            )

            ui: _BaseProgressUI
            if sys.stderr.isatty():
                ui = _SpinnerProgressUI(total_steps=total_steps)
            else:
                ui = _LineProgressUI(total_steps=total_steps)

            use_async = args.async_mode or (running_in_daemon and not args.foreground)

            plan_req: PlanRequest | None = None
            git_policy: GitPolicy | None = None

            if not use_async and sys.stdin.isatty():
                allow_auto = any(
                    (a.provider or engine.pipeline.provider).type != "deterministic"
                    for a in engine.pipeline.actors
                )
                plan_choices = ["none", "user"]
                if allow_auto:
                    plan_choices = ["auto", "user", "none"]
                default_plan = "auto" if allow_auto else "none"
                plan_mode = _prompt_choice(
                    f"Plan mode ({'/'.join(plan_choices)})",
                    choices=plan_choices,
                    default=default_plan,
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

                gd = engine.pipeline.git_defaults
                if gd is not None and gd.mode != "off":
                    git_policy = GitPolicy(
                        mode=gd.mode,
                        branch_prefix=gd.branch_prefix,
                        auto_commit=gd.auto_commit,
                        branch_template=gd.branch_template,
                        branch_slug=gd.branch_slug,
                        commit_message_template=gd.commit_message_template,
                    )

                # Interactive override (optional). By default, run uses pipeline.git_defaults.
                if args.ask_git or gd is None:
                    default_mode = "off"
                    if gd is not None:
                        default_mode = gd.mode
                    elif (workspace_dir.resolve() / ".git").exists():
                        default_mode = "branch"
                    git_mode = _prompt_choice(
                        "Git safety (branch/check/off)",
                        choices=["branch", "check", "off"],
                        default=default_mode,
                    )
                    if git_mode == "off":
                        git_policy = None
                    else:
                        branch_template = gd.branch_template if gd is not None else None
                        branch_slug = gd.branch_slug if gd is not None else None
                        commit_message_template = gd.commit_message_template if gd is not None else None
                        prefix = gd.branch_prefix if gd is not None else "orch/"
                        auto_commit = gd.auto_commit if gd is not None else False
                        if git_mode == "branch":
                            prefix = input(f"Git branch prefix [{prefix}]: ").strip() or prefix
                            auto_commit = (
                                _prompt_choice(
                                    "Auto-commit after implementer steps? (y/n)",
                                    choices=["y", "n"],
                                    default="y" if auto_commit else "n",
                                )
                                == "y"
                            )
                        git_policy = GitPolicy(
                            mode=git_mode,
                            branch_prefix=prefix,
                            auto_commit=auto_commit if git_mode == "branch" else False,
                            branch_template=branch_template,
                            branch_slug=branch_slug,
                            commit_message_template=commit_message_template,
                        )
            elif use_async:
                # Async mode currently uses pipeline defaults only and skips interactive prompts.
                gd = engine.pipeline.git_defaults
                if gd is not None and gd.mode != "off":
                    git_policy = GitPolicy(
                        mode=gd.mode,
                        branch_prefix=gd.branch_prefix,
                        auto_commit=gd.auto_commit,
                        branch_template=gd.branch_template,
                        branch_slug=gd.branch_slug,
                        commit_message_template=gd.commit_message_template,
                    )

            if args.async_mode and not running_in_daemon:
                print("ERROR: --async requested but no daemon is running.", file=sys.stderr)
                print(
                    "Start daemon with: python main.py daemon start --workspace ...",
                    file=sys.stderr,
                )
                return 2

            if use_async:
                cmd = daemon.submit_run(
                    task_id=args.task,
                    timeout_s_override=args.timeout_s,
                    idle_timeout_s_override=args.idle_timeout_s,
                )
                run_dir = daemon.runs_root / cmd.run_id
                print(f"Run queued: {cmd.run_id}")
                print(f"Dir:    {run_dir}")
                if args.wait:
                    state = _wait_for_daemon_run(
                        daemon,
                        run_id=cmd.run_id,
                        poll_interval=max(0.2, float(args.poll_interval)),
                        follow=True,
                        print_timeline=True,
                    )
                    return _print_final_for_run(cmd.run_id, run_dir, state)
                return 0

            outcome = engine.run(
                progress=ui.log_line,
                on_event=ui.on_event,
                plan=plan_req,
                git=git_policy,
                timeout_s_override=args.timeout_s,
                idle_timeout_s_override=args.idle_timeout_s,
            )
            ui.finish()
            print("")
            print(f"Run:    {outcome.run_id}")
            print(f"Status: {outcome.status.value}")
            print(f"Dir:    {outcome.run_dir}")
            print("")
            if outcome.status == Status.NEEDS_INPUT:
                ni_path = outcome.run_dir / "NEEDS_INPUT.md"
                if ni_path.exists():
                    print(f"NEEDS_INPUT file: {ni_path}")
                esc_path = outcome.run_dir / "escalation" / "escalation.md"
                if esc_path.exists():
                    print(f"Escalation file: {esc_path}")
                if ni_path.exists() or esc_path.exists():
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
