from __future__ import annotations

import argparse
import ast
import py_compile
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Issue:
    path: Path
    line: int | None
    code: str
    message: str

    def render(self) -> str:
        loc = str(self.path)
        if self.line is not None:
            loc = f"{loc}:{self.line}"
        return f"{loc}: {self.code}: {self.message}"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _iter_python_files(root: Path) -> list[Path]:
    paths: list[Path] = []
    paths.append(root / "main.py")
    for p in (root / "src").rglob("*.py"):
        paths.append(p)
    for p in (root / "tools").rglob("*.py"):
        paths.append(p)
    # Deterministic order.
    return sorted({p.resolve() for p in paths if p.is_file()})


def _check_text_conventions(path: Path, text: str) -> list[Issue]:
    issues: list[Issue] = []
    if "\r" in text:
        issues.append(
            Issue(path=path, line=None, code="EOL001", message="CRLF/CR line endings found")
        )

    if text and not text.endswith("\n"):
        issues.append(
            Issue(path=path, line=None, code="EOL002", message="File does not end with newline")
        )

    for idx, line in enumerate(text.splitlines(), start=1):
        if "\t" in line:
            issues.append(Issue(path=path, line=idx, code="WS001", message="Tab character found"))
        if line.rstrip(" \t") != line:
            issues.append(Issue(path=path, line=idx, code="WS002", message="Trailing whitespace"))

    # Ensure files stay ASCII-clean unless deliberately changed.
    try:
        text.encode("ascii")
    except UnicodeEncodeError:
        issues.append(
            Issue(path=path, line=None, code="ENC001", message="Non-ASCII characters found")
        )

    return issues


_RE_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _module_unused_imports(path: Path, text: str) -> list[Issue]:
    """
    Very small "lint" pass:
    - flags unused import names within the module based on AST Name loads.
    - treats type annotations as usage (AST still contains Name nodes).
    - ignores from __future__ imports.
    - ignores imports named '_' (commonly used as a deliberate placeholder).
    """

    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as e:
        return [
            Issue(path=path, line=e.lineno or None, code="SYN001", message=f"SyntaxError: {e.msg}")
        ]

    imported: dict[str, int] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                name = a.asname or a.name.split(".")[0]
                if _RE_IDENTIFIER.match(name):
                    imported.setdefault(name, node.lineno)
        elif isinstance(node, ast.ImportFrom):
            if node.module == "__future__":
                continue
            for a in node.names:
                if a.name == "*":
                    continue
                name = a.asname or a.name
                if _RE_IDENTIFIER.match(name):
                    imported.setdefault(name, node.lineno)

    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            used.add(node.id)

    # Honor __all__ for simple constant lists/tuples: __all__ = ["x", "y"]
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "__all__":
                    if isinstance(node.value, (ast.List, ast.Tuple)):
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                used.add(elt.value)

    issues: list[Issue] = []
    for name, lineno in sorted(imported.items(), key=lambda kv: (kv[1], kv[0])):
        if name == "_":
            continue
        if name not in used:
            issues.append(
                Issue(path=path, line=lineno, code="IMP001", message=f"Unused import: {name!r}")
            )
    return issues


def _compile_check(path: Path) -> list[Issue]:
    try:
        py_compile.compile(str(path), doraise=True)
        return []
    except py_compile.PyCompileError as e:
        return [Issue(path=path, line=None, code="CPL001", message=str(e))]


def _run_example_pipeline(root: Path) -> list[Issue]:
    """
    Runs the example workspace in a temporary directory so checks do not dirty the repo.
    """
    issues: list[Issue] = []
    example_workspace = root / "example" / "workspace"
    if not example_workspace.exists():
        return [
            Issue(
                path=example_workspace, line=None, code="EX001", message="Missing example workspace"
            )
        ]

    with tempfile.TemporaryDirectory(prefix="orch_check_") as td:
        tmp_ws = Path(td) / "workspace"
        shutil.copytree(example_workspace, tmp_ws)

        # Import project code by adding repo root to sys.path (no packaging required).
        sys.path.insert(0, str(root))
        try:
            from src.core.types import Status
            from src.orchestrator.engine import OrchestratorEngine
        finally:
            # Keep sys.path stable for callers.
            try:
                sys.path.remove(str(root))
            except ValueError:
                pass

        try:
            engine = OrchestratorEngine.load(workspace_dir=tmp_ws)
            outcome = engine.run(progress=None)
        except Exception as e:
            issues.append(
                Issue(
                    path=example_workspace,
                    line=None,
                    code="EX002",
                    message=f"Example run failed: {e}",
                )
            )
            return issues

        if outcome.status != Status.OK:
            issues.append(
                Issue(
                    path=example_workspace,
                    line=None,
                    code="EX003",
                    message=f"Example run status is {outcome.status.value!r}, expected 'OK'",
                )
            )

        # Basic invariant: run writes state + timeline.
        if not (outcome.run_dir / "state.json").exists():
            issues.append(
                Issue(path=outcome.run_dir, line=None, code="EX004", message="Missing state.json")
            )
        if not (outcome.run_dir / "timeline.jsonl").exists():
            issues.append(
                Issue(
                    path=outcome.run_dir, line=None, code="EX005", message="Missing timeline.jsonl"
                )
            )
        if not (outcome.run_dir / "final.txt").exists():
            issues.append(
                Issue(path=outcome.run_dir, line=None, code="EX006", message="Missing final.txt")
            )

    return issues


def _check_codex_jsonl_extraction(root: Path) -> list[Issue]:
    """
    Unit-style check: Codex JSONL often emits the final assistant content as:
        {"type":"item.completed","item":{"type":"agent_message","text":"..."}}
    Ensure we extract that correctly.
    """
    sys.path.insert(0, str(root))
    try:
        from src.providers.codex_cli import _extract_final_agent_message
    finally:
        try:
            sys.path.remove(str(root))
        except ValueError:
            pass

    events = [
        {"type": "thread.started", "thread_id": "t"},
        {"type": "turn.started"},
        {
            "type": "item.completed",
            "item": {"id": "item_0", "type": "agent_message", "text": "HELLO"},
        },
        {"type": "turn.completed"},
    ]
    final_text, _source = _extract_final_agent_message(events, raw_stdout="")
    if final_text != "HELLO":
        return [
            Issue(
                path=root / "src" / "providers" / "codex_cli.py",
                line=None,
                code="CODEX001",
                message=f"Expected 'HELLO' from item.completed extraction, got: {final_text!r}",
            )
        ]
    return []


def _run_external_tool(
    *,
    root: Path,
    cmd: list[str],
    code: str,
    missing_code: str,
    missing_message: str,
) -> list[Issue]:
    exe = shutil.which(cmd[0])
    if exe is None:
        return [Issue(path=root, line=None, code=missing_code, message=missing_message)]

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(root),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except Exception as e:
        return [Issue(path=root, line=None, code=code, message=f"Failed to run {cmd[0]!r}: {e}")]

    if proc.returncode == 0:
        return []

    # Keep output short for signal; full output is still available in stdout if needed.
    out = (proc.stdout or "").strip()
    head = "\n".join(out.splitlines()[:40])
    msg = f"{cmd[0]} failed with exit code {proc.returncode}"
    if head:
        msg += f":\n{head}"
    return [Issue(path=root, line=None, code=code, message=msg)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Project checks (stdlib-only).")
    ap.add_argument("--no-example", action="store_true", help="Skip running the example pipeline.")
    ap.add_argument("--with-ruff", action="store_true", help="Also run ruff (if installed).")
    ap.add_argument(
        "--with-black", action="store_true", help="Also run black --check (if installed)."
    )
    args = ap.parse_args(argv)

    root = _project_root()
    paths = _iter_python_files(root)

    all_issues: list[Issue] = []
    for p in paths:
        try:
            text = p.read_text(encoding="utf-8")
        except Exception as e:
            all_issues.append(
                Issue(path=p, line=None, code="IO001", message=f"Failed to read: {e}")
            )
            continue
        all_issues.extend(_check_text_conventions(p, text))
        all_issues.extend(_compile_check(p))
        all_issues.extend(_module_unused_imports(p, text))

    all_issues.extend(_check_codex_jsonl_extraction(root))

    if not args.no_example:
        all_issues.extend(_run_example_pipeline(root))

    if args.with_ruff:
        all_issues.extend(
            _run_external_tool(
                root=root,
                cmd=["ruff", "check", "main.py", "src", "tools"],
                code="RUFF001",
                missing_code="RUFF000",
                missing_message="ruff not found on PATH (install ruff or omit --with-ruff)",
            )
        )

    if args.with_black:
        all_issues.extend(
            _run_external_tool(
                root=root,
                cmd=["black", "--check", "--diff", "main.py", "src", "tools"],
                code="BLK001",
                missing_code="BLK000",
                missing_message="black not found on PATH (install black or omit --with-black)",
            )
        )

    if all_issues:
        for it in sorted(all_issues, key=lambda i: (str(i.path), i.line or 0, i.code)):
            print(it.render(), file=sys.stderr)
        print(f"\nFAIL: {len(all_issues)} issue(s)", file=sys.stderr)
        return 1

    print("OK: all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
