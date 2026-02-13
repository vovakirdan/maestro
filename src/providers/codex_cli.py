from __future__ import annotations

import json
import os
import selectors
import subprocess
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from src.core.types import JsonDict
from src.providers.base import ProviderError, ProviderResult


def _now_iso_utc() -> str:
    # Avoid datetime to keep this tiny and monotonic-based timing separate.
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _write_jsonl(path: Path, events: Iterable[JsonDict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev, sort_keys=True) + "\n")


def _extract_final_agent_message(events: Sequence[JsonDict], raw_stdout: str) -> tuple[str, str]:
    """
    Attempt to extract the final assistant message from parsed JSONL events.
    Returns (final_text, source_label).
    """

    def candidate_from_obj(obj: object) -> str | None:
        if not isinstance(obj, dict):
            return None

        # Ignore likely token-delta events.
        t = obj.get("type")
        if isinstance(t, str) and t.lower() in {"token", "delta"}:
            return None
        if "delta" in obj:
            return None

        for k in ("agent_message", "final_text", "final", "message", "content", "text"):
            v = obj.get(k)
            if isinstance(v, str) and v.strip():
                return v

        data = obj.get("data")
        if isinstance(data, dict):
            return candidate_from_obj(data)
        return None

    final_preferred: str | None = None
    last_any: str | None = None

    for ev in events:
        if not isinstance(ev, dict):
            continue
        cand = candidate_from_obj(ev)
        if cand:
            last_any = cand
            t = ev.get("type")
            if isinstance(t, str) and t.lower() in {"final", "agent_message", "assistant_message"}:
                final_preferred = cand

    if final_preferred is not None:
        return final_preferred, "events:type=final-like"
    if last_any is not None:
        return last_any, "events:last-candidate"
    if raw_stdout.strip():
        return raw_stdout.strip(), "stdout:fallback"
    return "", "empty"


@dataclass(frozen=True)
class CodexCLIProvider:
    """
    Provider that shells out to a local Codex CLI and parses JSONL events from stdout.

    The exact CLI flags are intentionally configurable via `command`.
    This provider assumes stdout is JSONL (one JSON object per line).
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
        timed_out = False
        idle_timed_out = False
        kill_sent = False
        forced_shutdown_break = False
        shutdown_started: float | None = None

        parsed_events: list[JsonDict] = []
        stdout_lines: list[str] = []
        stderr_lines: list[str] = []

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
            raise ProviderError(f"Codex CLI command not found: {cmd[0]!r}") from e

        assert proc.stdin is not None
        assert proc.stdout is not None
        assert proc.stderr is not None

        # Send prompt then close stdin so the process can proceed deterministically.
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

            # Escalate to SIGKILL if SIGTERM doesn't stop the process promptly.
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

            # Keep the select timeout short so timeouts are enforced precisely.
            wait_s = 0.2
            for key, _mask in sel.select(timeout=wait_s):
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
                    stripped = line.strip()
                    if stripped:
                        try:
                            obj = json.loads(stripped)
                            if isinstance(obj, dict):
                                parsed_events.append(obj)
                            else:
                                parsed_events.append({"type": "non_dict_event", "value": obj})
                        except json.JSONDecodeError:
                            parsed_events.append({"type": "non_json_stdout", "text": stripped})
                else:
                    stderr_lines.append(line)

            if proc.poll() is not None and not sel.get_map():
                break

        try:
            sel.close()
        except Exception:
            pass

        # Ensure the process is gone; never hang indefinitely here.
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
        _write_jsonl(artifacts_dir / "provider_events.jsonl", parsed_events)

        final_text, final_source = _extract_final_agent_message(parsed_events, raw_stdout)

        metadata: JsonDict = {
            "provider": "codex_cli",
            "started_at_utc": started_at_utc,
            "exit_code": exit_code,
            "duration_s": round(time.monotonic() - started, 6),
            "timeout_s": timeout_s,
            "idle_timeout_s": idle_timeout_s,
            "timed_out": timed_out,
            "idle_timed_out": idle_timed_out,
            "kill_sent": kill_sent,
            "forced_shutdown_break": forced_shutdown_break,
            "final_text_source": final_source,
        }
        (artifacts_dir / "provider_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (artifacts_dir / "provider_final.txt").write_text(final_text, encoding="utf-8")

        return ProviderResult(final_text=final_text, events=parsed_events, metadata=metadata)
