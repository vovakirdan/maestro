from __future__ import annotations

import json
import os
import selectors
import subprocess
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from src.core.types import JsonDict
from src.providers.base import ProviderError, ProviderResult


def _now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _write_jsonl(path: Path, events: list[JsonDict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev, sort_keys=True) + "\n")


@dataclass(frozen=True)
class ClaudeCLIProvider:
    """
    Provider that shells out to a local Claude Code CLI.

    This is intentionally configurable via `command`. The provider writes `prompt`
    to stdin and captures stdout as the final response.

    Recommended non-interactive command (example):
        claude -p --output-format text --input-format text --permission-mode acceptEdits
    """

    command: tuple[str, ...]
    cwd: Path | None = None
    extra_env: Mapping[str, str] | None = None
    terminate_grace_s: float = 3.0

    def run(
        self,
        prompt: str,
        *,
        artifacts_dir: Path,
        timeout_s: float,
        idle_timeout_s: float,
        on_event: Callable[[JsonDict], None] | None = None,
    ) -> ProviderResult:
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        cmd = list(self.command)
        env = os.environ.copy()
        if self.extra_env:
            env.update(dict(self.extra_env))

        started_at_utc = _now_iso_utc()
        (artifacts_dir / "provider_command.json").write_text(
            json.dumps({"command": cmd}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        started = time.monotonic()
        last_output = started
        last_heartbeat = started
        timed_out = False
        idle_timed_out = False
        kill_sent = False
        forced_shutdown_break = False
        shutdown_started: float | None = None

        events: list[JsonDict] = []
        stdout_lines: list[str] = []
        stderr_lines: list[str] = []

        def deliver(ev: JsonDict) -> None:
            events.append(ev)
            if on_event is None:
                return
            try:
                on_event(ev)
            except Exception:
                pass

        deliver({"type": "provider", "provider": "claude_cli", "event": "start"})

        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(self.cwd) if self.cwd else None,
                env=env,
                text=True,
                bufsize=1,
            )
        except FileNotFoundError as e:
            raise ProviderError(f"Claude CLI command not found: {cmd[0]!r}") from e

        assert proc.stdin is not None
        assert proc.stdout is not None
        assert proc.stderr is not None

        try:
            proc.stdin.write(prompt)
            proc.stdin.flush()
        finally:
            try:
                proc.stdin.close()
            except Exception:
                pass

        sel = selectors.DefaultSelector()
        sel.register(proc.stdout, selectors.EVENT_READ, data="stdout")
        sel.register(proc.stderr, selectors.EVENT_READ, data="stderr")

        while True:
            now = time.monotonic()
            hard_timeout_hit = timeout_s > 0 and (now - started) > timeout_s
            idle_timeout_hit = idle_timeout_s > 0 and (now - last_output) > idle_timeout_s

            if (hard_timeout_hit or idle_timeout_hit) and shutdown_started is None:
                shutdown_started = now
                timed_out = timed_out or hard_timeout_hit
                idle_timed_out = idle_timed_out or idle_timeout_hit
                try:
                    proc.terminate()
                except Exception:
                    pass

            if shutdown_started is not None and proc.poll() is None:
                if (now - shutdown_started) > self.terminate_grace_s and not kill_sent:
                    try:
                        proc.kill()
                        kill_sent = True
                    except Exception:
                        pass
                if kill_sent and (now - shutdown_started) > (self.terminate_grace_s + 1.0):
                    forced_shutdown_break = True
                    break

            if proc.poll() is not None and not sel.get_map():
                break

            for key, _mask in sel.select(timeout=0.2):
                kind = key.data
                stream = key.fileobj
                line = stream.readline()
                if line == "":
                    try:
                        sel.unregister(stream)
                    except Exception:
                        pass
                    continue

                last_output = time.monotonic()
                if kind == "stdout":
                    stdout_lines.append(line)
                else:
                    stderr_lines.append(line)

            now = time.monotonic()
            if on_event is not None and proc.poll() is None:
                if (now - last_output) >= 1.0 and (now - last_heartbeat) >= 1.0:
                    deliver(
                        {
                            "type": "heartbeat",
                            "elapsed_s": round(now - started, 1),
                            "idle_s": round(now - last_output, 1),
                        }
                    )
                    last_heartbeat = now

        try:
            sel.close()
        except Exception:
            pass

        try:
            proc.wait(timeout=0.2)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
                kill_sent = True
            except Exception:
                pass
            try:
                proc.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                forced_shutdown_break = True

        exit_code = proc.poll()
        raw_stdout = "".join(stdout_lines)
        raw_stderr = "".join(stderr_lines)

        (artifacts_dir / "provider_stdout.log").write_text(raw_stdout, encoding="utf-8")
        (artifacts_dir / "provider_stderr.log").write_text(raw_stderr, encoding="utf-8")
        _write_jsonl(artifacts_dir / "provider_events.jsonl", events)

        final_text = raw_stdout.strip()
        if final_text:
            final_text += "\n"

        deliver(
            {
                "type": "provider",
                "provider": "claude_cli",
                "event": "finish",
                "exit_code": exit_code,
            }
        )

        metadata: JsonDict = {
            "provider": "claude_cli",
            "started_at_utc": started_at_utc,
            "exit_code": exit_code,
            "duration_s": round(time.monotonic() - started, 6),
            "timeout_s": timeout_s,
            "idle_timeout_s": idle_timeout_s,
            "timed_out": timed_out,
            "idle_timed_out": idle_timed_out,
            "kill_sent": kill_sent,
            "forced_shutdown_break": forced_shutdown_break,
            "final_text_source": "stdout",
        }
        (artifacts_dir / "provider_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (artifacts_dir / "provider_final.txt").write_text(final_text, encoding="utf-8")

        return ProviderResult(final_text=final_text, events=events, metadata=metadata)

