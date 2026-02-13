from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.core.types import Status
from src.orchestrator.engine import OrchestratorEngine
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
            engine = OrchestratorEngine.load(workspace_dir=workspace_dir)
            outcome = engine.run(progress=print)
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
