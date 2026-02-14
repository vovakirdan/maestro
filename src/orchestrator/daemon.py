from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.core.runtime import generate_run_id
from src.core.types import JsonDict
from src.orchestrator.engine import OrchestratorEngine

DEFAULT_DAEMON_DATA_DIR_ENV = "ORCHESTRATOR_DAEMON_DIR"
DEFAULT_DAEMON_DATA_DIR = Path.home() / ".cache" / "orchestrator"
DEFAULT_POLL_INTERVAL = 1.0


def _now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _parse_iso_to_epoch(value: str) -> float:
    try:
        return time.mktime(time.strptime(value, "%Y-%m-%dT%H:%M:%SZ"))
    except Exception:
        return time.time()


def resolve_daemon_data_dir(raw: str | None = None) -> Path:
    if raw:
        return Path(raw).expanduser()
    env = os.environ.get(DEFAULT_DAEMON_DATA_DIR_ENV)
    if env:
        return Path(env).expanduser()
    return DEFAULT_DAEMON_DATA_DIR


def _read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return dict(default)
    try:
        parsed = json.loads(raw)
    except Exception:
        return dict(default)
    if not isinstance(parsed, dict):
        return dict(default)
    return parsed


def _write_json(path: Path, obj: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, event: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def _read_tail_lines(path: Path, limit: int) -> list[str]:
    if not path.exists():
        return []
    if limit <= 0:
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return lines[-limit:]


def workspace_key(workspace_dir: Path) -> str:
    return hashlib.sha256(str(workspace_dir.resolve()).encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class OrchestratorDaemonConfig:
    workspace_dir: Path
    daemon_data_dir: Path
    poll_interval_s: float = DEFAULT_POLL_INTERVAL


@dataclass(frozen=True)
class RunDispatchCommand:
    run_id: str
    task_id: str | None
    timeout_s_override: float | None
    idle_timeout_s_override: float | None

    def to_dict(self) -> JsonDict:
        return {
            "type": "run",
            "run_id": self.run_id,
            "task_id": self.task_id,
            "timeout_s_override": self.timeout_s_override,
            "idle_timeout_s_override": self.idle_timeout_s_override,
            "submitted_at_utc": _now_iso_utc(),
        }

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> "RunDispatchCommand":
        if not isinstance(raw, dict):
            raise ValueError("run command must be a dict")
        if raw.get("type") != "run":
            raise ValueError("unsupported command type")

        run_id = raw.get("run_id")
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id must be a non-empty string")

        task_id = raw.get("task_id")
        if task_id is not None and (not isinstance(task_id, str) or not task_id.strip()):
            raise ValueError("task_id must be string")

        to_override = raw.get("timeout_s_override")
        if to_override is None:
            timeout_s: float | None = None
        else:
            timeout_s = float(to_override)

        io_override = raw.get("idle_timeout_s_override")
        if io_override is None:
            idle_timeout_s: float | None = None
        else:
            idle_timeout_s = float(io_override)

        return RunDispatchCommand(
            run_id=run_id.strip(),
            task_id=task_id.strip() if isinstance(task_id, str) else None,
            timeout_s_override=timeout_s,
            idle_timeout_s_override=idle_timeout_s,
        )


class RunStatus:
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    OK = "OK"
    FAILED = "FAILED"
    NEEDS_INPUT = "NEEDS_INPUT"


@dataclass(frozen=True)
class DaemonRunRecord:
    run_id: str
    status: str
    workspace_key: str
    workspace_dir: str
    task_id: str | None
    created_at_utc: str | None
    finished_at_utc: str | None
    pipeline_path: str | None
    run_dir: str

    @staticmethod
    def from_state(
        run_id: str,
        workspace_key: str,
        workspace_dir: str,
        state: JsonDict,
        run_dir: Path,
    ) -> "DaemonRunRecord":
        status = state.get("status", "UNKNOWN")
        if status == "RUNNING":
            status = RunStatus.RUNNING
        return DaemonRunRecord(
            run_id=run_id,
            status=str(status),
            workspace_key=workspace_key,
            workspace_dir=workspace_dir,
            task_id=state.get("task_id") if isinstance(state.get("task_id"), str) else None,
            created_at_utc=state.get("created_at_utc") if isinstance(state.get("created_at_utc"), str) else None,
            finished_at_utc=state.get("finished_at_utc") if isinstance(state.get("finished_at_utc"), str) else None,
            pipeline_path=state.get("pipeline_path") if isinstance(state.get("pipeline_path"), str) else None,
            run_dir=run_dir.as_posix(),
        )


class OrchestratorDaemon:
    """Workspace-scoped daemon manager for orchestrated runs."""

    def __init__(
        self,
        *,
        workspace_dir: Path,
        daemon_data_dir: Path,
        poll_interval_s: float = DEFAULT_POLL_INTERVAL,
    ) -> None:
        self.workspace_dir = workspace_dir.expanduser().resolve()
        self.daemon_data_dir = daemon_data_dir.expanduser()
        self.poll_interval_s = max(0.2, float(poll_interval_s))

        self._workspace_key = workspace_key(self.workspace_dir)
        self.workspace_root = self.daemon_data_dir / "daemon" / self._workspace_key
        self.meta_path = self.workspace_root / "meta.json"
        self.queue_path = self.workspace_root / "queue.jsonl"
        self.queue_state_path = self.workspace_root / "queue.state"
        self.runs_root = self.workspace_root / "runs"
        self.stop_path = self.workspace_root / "stop.request"
        self.registry_path = self.daemon_data_dir / "registry.json"
        self._running = False

    @property
    def workspace_key(self) -> str:
        return self._workspace_key

    @property
    def workspace_dir_s(self) -> str:
        return self.workspace_dir.as_posix()

    def _ensure_layout(self) -> None:
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.runs_root.mkdir(parents=True, exist_ok=True)
        if not self.queue_path.exists():
            self.queue_path.write_text("", encoding="utf-8")

    @staticmethod
    def _read_jsonl_chunk(path: Path, start_offset: int) -> tuple[list[str], int]:
        if not path.exists():
            return [], 0
        try:
            st = path.stat()
        except OSError:
            return [], 0
        if st.st_size < start_offset:
            start_offset = 0

        with path.open("rb") as handle:
            handle.seek(start_offset)
            data = handle.read()
            new_offset = handle.tell()

        if not data:
            return [], new_offset

        if data[-1:] != b"\n":
            lines = data.rsplit(b"\n", 1)
            if len(lines) == 1:
                return [], new_offset
            data = lines[0] + b"\n"
            new_offset -= len(lines[1])

        out: list[str] = []
        for raw in data.splitlines():
            if not raw.strip():
                continue
            try:
                out.append(raw.decode("utf-8"))
            except Exception:
                continue
        return out, new_offset

    def _read_queue_offset(self) -> int:
        data = _read_json(self.queue_state_path, {"offset": 0})
        value = data.get("offset")
        if not isinstance(value, int) or value < 0:
            return 0
        return value

    def _write_queue_offset(self, offset: int) -> None:
        _write_json(self.queue_state_path, {"offset": int(offset)})

    def _append_queue_line(self, line: str) -> None:
        self.queue_path.parent.mkdir(parents=True, exist_ok=True)
        with self.queue_path.open("a", encoding="utf-8") as handle:
            handle.write(line.rstrip("\n") + "\n")

    def _load_registry(self) -> dict[str, JsonDict]:
        data = _read_json(self.registry_path, {"workspaces": []})
        workspaces = data.get("workspaces")
        if not isinstance(workspaces, list):
            workspaces = []

        out: dict[str, JsonDict] = {}
        for entry in workspaces:
            if not isinstance(entry, dict):
                continue
            key = entry.get("workspace_key")
            if isinstance(key, str) and key:
                out[key] = entry
        return out

    def _save_registry(self, payload: dict[str, JsonDict]) -> None:
        entries = sorted(payload.values(), key=lambda x: str(x.get("workspace_key")))
        _write_json(self.registry_path, {"workspaces": entries})

    def _upsert_registry_entry(self) -> None:
        entries = self._load_registry()
        entries[self.workspace_key] = {
            "workspace_key": self.workspace_key,
            "workspace_dir": self.workspace_dir_s,
            "last_seen_utc": _now_iso_utc(),
        }
        self._save_registry(entries)

    def _read_meta(self) -> JsonDict:
        return _read_json(
            self.meta_path,
            {
                "workspace_dir": self.workspace_dir_s,
                "workspace_key": self.workspace_key,
                "daemon_pid": None,
                "active_run_id": None,
                "created_at_utc": None,
                "updated_at_utc": None,
                "heartbeat_ts": None,
                "task_id": None,
            },
        )

    def _write_meta(self, updates: dict[str, Any]) -> None:
        meta = self._read_meta()
        meta.update(updates)
        meta["updated_at_utc"] = _now_iso_utc()
        _write_json(self.meta_path, meta)

    def is_running(self) -> tuple[bool, int | None]:
        meta = self._read_meta()
        pid_raw = meta.get("daemon_pid")
        if not isinstance(pid_raw, int):
            return False, None
        try:
            os.kill(pid_raw, 0)
        except Exception:
            return False, pid_raw
        return True, pid_raw

    def heartbeat(self) -> None:
        now = _now_iso_utc()
        self._write_meta({"heartbeat_ts": now})

    def start(self) -> int:
        self._ensure_layout()
        self._upsert_registry_entry()

        running, pid = self.is_running()
        if running:
            raise RuntimeError(f"daemon already running for workspace (pid={pid})")

        if self.stop_path.exists():
            self.stop_path.unlink()

        cmd = [
            sys.executable,
            str(Path(sys.argv[0]).resolve()),
            "daemon",
            "worker",
            "--workspace",
            self.workspace_dir.as_posix(),
            "--daemon-data-dir",
            self.daemon_data_dir.as_posix(),
            "--poll-interval",
            str(self.poll_interval_s),
        ]
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        self._write_meta(
            {
                "workspace_dir": self.workspace_dir_s,
                "workspace_key": self.workspace_key,
                "daemon_pid": proc.pid,
                "active_run_id": None,
                "created_at_utc": _now_iso_utc(),
                "heartbeat_ts": _now_iso_utc(),
            }
        )
        return proc.pid

    def stop(self, *, force: bool = False) -> bool:
        running, pid = self.is_running()
        if not running:
            return False

        self.stop_path.write_text(_now_iso_utc(), encoding="utf-8")
        if pid is None:
            return False

        try:
            os.kill(pid, signal.SIGTERM)
        except Exception:
            return False

        started = time.monotonic()
        while time.monotonic() - started < 5.0:
            if not self.is_running()[0]:
                return True
            time.sleep(0.1)

        if force:
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:
                return False
            return True

        return False

    def status(self) -> JsonDict:
        running, pid = self.is_running()
        meta = self._read_meta()
        heartbeat = meta.get("heartbeat_ts")
        out: JsonDict = {
            "workspace_dir": self.workspace_dir_s,
            "workspace_key": self.workspace_key,
            "running": running,
            "pid": pid,
            "active_run_id": meta.get("active_run_id"),
            "heartbeat_ts": heartbeat,
            "created_at_utc": meta.get("created_at_utc"),
            "updated_at_utc": meta.get("updated_at_utc"),
            "meta_path": self.meta_path.as_posix(),
            "queue_path": self.queue_path.as_posix(),
        }
        if isinstance(heartbeat, str):
            out["heartbeat_age_s"] = max(0.0, time.time() - _parse_iso_to_epoch(heartbeat))
        return out

    def submit_run(
        self,
        *,
        task_id: str | None,
        timeout_s_override: float | None,
        idle_timeout_s_override: float | None,
    ) -> RunDispatchCommand:
        self._ensure_layout()
        cmd = RunDispatchCommand(
            run_id=generate_run_id(),
            task_id=task_id,
            timeout_s_override=timeout_s_override,
            idle_timeout_s_override=idle_timeout_s_override,
        )
        self._append_queue_line(json.dumps(cmd.to_dict(), sort_keys=True))

        run_dir = self.runs_root / cmd.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            run_dir / "state.json",
            {
                "run_id": cmd.run_id,
                "status": RunStatus.PENDING,
                "task_id": cmd.task_id,
                "created_at_utc": _now_iso_utc(),
                "updated_at_utc": _now_iso_utc(),
                "steps": [],
                "current": {
                    "actor_id": None,
                    "step_name": None,
                    "attempt": None,
                    "preset": None,
                    "is_retry": False,
                },
                "progress": {
                    "invocation_index": 0,
                    "attempts_done": 0,
                    "max_steps": None,
                    "total_actor_count": None,
                },
                "heartbeat": {
                    "updated_at_utc": _now_iso_utc(),
                    "last_log_line_ts": time.time(),
                },
                "last_event": {
                    "type": "run_queued",
                    "ts_utc": _now_iso_utc(),
                },
            },
        )
        self._upsert_registry_entry()
        return cmd

    def _read_pending_commands(self) -> list[RunDispatchCommand]:
        offset = self._read_queue_offset()
        lines, new_offset = self._read_jsonl_chunk(self.queue_path, offset)
        if new_offset != offset:
            self._write_queue_offset(new_offset)

        out: list[RunDispatchCommand] = []
        for raw in lines:
            try:
                payload = json.loads(raw)
            except Exception:
                continue
            try:
                out.append(RunDispatchCommand.from_dict(payload))
            except Exception:
                continue
        return out

    def list_runs(
        self,
        *,
        task: str | None = None,
        status_filter: str | None = None,
        limit: int = 20,
    ) -> list[DaemonRunRecord]:
        if not self.runs_root.exists():
            return []

        run_dirs = sorted(
            [p for p in self.runs_root.iterdir() if p.is_dir() and p.name != "__pycache"],
            key=lambda p: p.name,
            reverse=True,
        )
        out: list[DaemonRunRecord] = []
        for run_dir in run_dirs:
            state = _read_json(run_dir / "state.json", {})
            rec = DaemonRunRecord.from_state(
                run_id=run_dir.name,
                workspace_key=self.workspace_key,
                workspace_dir=self.workspace_dir_s,
                state=state,
                run_dir=run_dir,
            )
            if task is not None and rec.task_id != task:
                continue
            if status_filter is not None and rec.status != status_filter:
                continue
            out.append(rec)
            if len(out) >= limit:
                break
        return out

    def read_run_state(self, run_id: str) -> JsonDict:
        return _read_json(self.runs_root / run_id / "state.json", {})

    def read_run_timeline(self, run_id: str, *, limit: int | None = None) -> list[str]:
        path = self.runs_root / run_id / "timeline.jsonl"
        if not path.exists():
            return []
        if limit is None or limit <= 0:
            return path.read_text(encoding="utf-8").splitlines()
        return _read_tail_lines(path, limit=limit)

    def run_worker(self) -> None:
        self._ensure_layout()
        self._upsert_registry_entry()
        self._write_meta(
            {
                "workspace_dir": self.workspace_dir_s,
                "workspace_key": self.workspace_key,
                "daemon_pid": os.getpid(),
                "active_run_id": None,
            }
        )
        self._running = True
        try:
            self._worker_loop()
        finally:
            self._write_meta({"daemon_pid": None, "active_run_id": None})

    def _touch_run_state(self, run_id: str, values: JsonDict) -> None:
        path = self.runs_root / run_id / "state.json"
        state = _read_json(path, {})
        state.update(values)
        state["run_id"] = run_id
        state["updated_at_utc"] = _now_iso_utc()
        heartbeat = state.get("heartbeat")
        if not isinstance(heartbeat, dict):
            heartbeat = {}
        heartbeat["updated_at_utc"] = _now_iso_utc()
        heartbeat["last_log_line_ts"] = time.time()
        state["heartbeat"] = heartbeat
        _write_json(path, state)

    def _worker_loop(self) -> None:
        last_poll = 0.0
        while self._running:
            if self.stop_path.exists():
                break
            if (time.monotonic() - last_poll) < self.poll_interval_s:
                time.sleep(0.05)
                continue

            last_poll = time.monotonic()
            self.heartbeat()
            self._upsert_registry_entry()

            meta = self._read_meta()
            if meta.get("active_run_id"):
                continue

            commands = self._read_pending_commands()
            if not commands:
                continue

            for cmd in commands:
                if self.stop_path.exists():
                    break
                self._run_command(cmd)

        # keep observability after stop/shutdown
        self.heartbeat()

    def _run_command(self, cmd: RunDispatchCommand) -> None:
        run_id = cmd.run_id
        run_dir = self.runs_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        timeline_path = run_dir / "timeline.jsonl"

        self._write_meta({"active_run_id": run_id})
        _append_jsonl(
            timeline_path,
            {
                "type": "run_dequeued",
                "run_id": run_id,
                "task_id": cmd.task_id,
            },
        )
        self._touch_run_state(
            run_id,
            {
                "status": RunStatus.RUNNING,
                "task_id": cmd.task_id,
                "error": None,
            },
        )

        try:
            engine = OrchestratorEngine.load(workspace_dir=self.workspace_dir, task_id=cmd.task_id)
            self._touch_run_state(
                run_id,
                {
                    "pipeline_path": (engine.orchestrator_dir / "pipeline.json").as_posix(),
                },
            )

            outcome = engine.run(
                progress=None,
                on_event=None,
                timeout_s_override=cmd.timeout_s_override,
                idle_timeout_s_override=cmd.idle_timeout_s_override,
                run_root_override=self.runs_root,
                run_id_override=run_id,
                emit_state_events=True,
                on_event_for_state=lambda event: _append_jsonl(
                    timeline_path,
                    {
                        "type": "daemon_state",
                        "run_id": run_id,
                        "payload": event,
                    },
                ),
            )

            self._touch_run_state(
                run_id,
                {
                    "status": outcome.status.value,
                    "task_id": cmd.task_id,
                },
            )

            _append_jsonl(
                timeline_path,
                {
                    "type": "run_finished",
                    "run_id": run_id,
                    "status": outcome.status.value,
                },
            )
        except Exception as exc:
            self._touch_run_state(
                run_id,
                {
                    "status": RunStatus.FAILED,
                    "error": str(exc),
                },
            )
            _append_jsonl(
                timeline_path,
                {
                    "type": "run_failed",
                    "run_id": run_id,
                    "error": str(exc),
                },
            )
            (run_dir / "NEEDS_INPUT.md").write_text(
                "NEEDS_INPUT\n\nDaemon execution failed.\n", encoding="utf-8"
            )
        finally:
            self._write_meta({"active_run_id": None})


def render_status_lines(status: JsonDict) -> list[str]:
    out = [
        f"workspace: {status.get('workspace_dir', '')}",
        f"workspace_key: {status.get('workspace_key', '')}",
        f"running: {status.get('running', False)}",
        f"pid: {status.get('pid')}",
        f"active_run_id: {status.get('active_run_id')}",
        f"heartbeat: {status.get('heartbeat_ts')}",
    ]
    if status.get("heartbeat_age_s") is not None:
        out.append(f"heartbeat_age_s: {status.get('heartbeat_age_s'):.1f}")
    return out
