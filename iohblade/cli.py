"""The ``iohblade`` command line interface.

Commands:
    iohblade validate experiment.yml   Check a configuration without running it.
    iohblade run experiment.yml        Run (or resume) an experiment.
    iohblade list [kind]               List built-in components.
    iohblade schema [-o file]          Print the JSON schema of the YAML format.

Exit codes: 0 success, 1 runs failed, 2 invalid configuration or usage.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys

EXIT_OK = 0
EXIT_RUN_FAILED = 1
EXIT_CONFIG = 2

_KINDS = {
    "problems": "problem",
    "suites": "suite",
    "methods": "method",
    "llms": "llm",
    "loggers": "logger",
}


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("config", help="Path of the YAML experiment file.")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override a value, e.g. --set experiment.runs=1 or "
            "--set methods.0.budget=20 (repeatable)."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show output printed while the components are created.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="iohblade",
        description="Define, check and run BLADE experiments from YAML files.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("validate", help="Check a configuration without running it.")
    _add_config_args(p)

    p = sub.add_parser("run", help="Run (or resume) an experiment.")
    _add_config_args(p)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show the planned runs; do not run anything.",
    )

    p = sub.add_parser("list", help="List the built-in components.")
    p.add_argument(
        "kind",
        nargs="?",
        choices=sorted(_KINDS),
        help="Only list this kind of component.",
    )
    p.add_argument(
        "--check",
        action="store_true",
        help="Also check whether each component can be imported.",
    )

    p = sub.add_parser("schema", help="Print the JSON schema of the YAML format.")
    p.add_argument("-o", "--output", help="Write the schema to this file.")
    return parser


def _load_plan(args):
    from .config import load_plan

    if args.verbose:
        return load_plan(args.config, overrides=args.overrides)
    # Some components print while they are created; keep the output clean.
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        return load_plan(args.config, overrides=args.overrides)


def _cmd_validate(args) -> int:
    plan = _load_plan(args)
    print(plan.summary())
    print("\nConfiguration is valid.")
    return EXIT_OK


def _cmd_run(args) -> int:
    plan = _load_plan(args)
    if args.dry_run:
        print(plan.summary())
        return EXIT_OK
    experiment = plan.make_experiment()
    try:
        experiment()
    except RuntimeError as e:
        print(f"\n{e}", file=sys.stderr)
        return EXIT_RUN_FAILED
    print(f"\nResults: {plan.output_dir}")
    return EXIT_OK


def _cmd_list(args) -> int:
    from .registry import registry

    kinds = [args.kind] if args.kind else list(_KINDS)
    for i, kind in enumerate(kinds):
        if i:
            print()
        print(f"{kind}:")
        entries = registry.entries(_KINDS[kind])
        width = max((len(e.key) for e in entries), default=0)
        for e in entries:
            status = ""
            if args.check:
                status = "  [ok]" if e.is_available() else "  [missing dependencies]"
            hint = f" (dependencies: {e.install_hint()})" if e.install_hint() else ""
            print(f"  {e.key:<{width}}  {e.description}{hint}{status}")
    return EXIT_OK


def _cmd_schema(args) -> int:
    from .config import json_schema

    text = json.dumps(json_schema(), indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(text + "\n")
        print(f"Schema written to {args.output}")
    else:
        print(text)
    return EXIT_OK


_COMMANDS = {
    "validate": _cmd_validate,
    "run": _cmd_run,
    "list": _cmd_list,
    "schema": _cmd_schema,
}


def main(argv: list[str] | None = None) -> int:
    """Entry point of the ``iohblade`` command."""
    from .config import ConfigError

    args = build_parser().parse_args(argv)
    try:
        return _COMMANDS[args.command](args)
    except ConfigError as e:
        print(str(e), file=sys.stderr)
        return EXIT_CONFIG
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
