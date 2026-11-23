"""The ``iohblade`` command line interface.

Commands:
    iohblade init [dir]                Create a starter experiment.
    iohblade validate experiment.yml   Check a configuration without running it.
    iohblade check experiment.yml      Smoke test: environments, evaluation,
                                       methods (no LLM calls unless --llm).
    iohblade run experiment.yml        Run (or resume) an experiment.
    iohblade list [kind]               List built-in components.
    iohblade schema [-o file]          Print the JSON schema of the YAML format.
    iohblade cache list|clear          Show or delete cached evaluation
                                       environments.
    iohblade skill [dir]               Install the BLADE skill for coding
                                       agents (default: .claude/skills).

Exit codes: 0 success, 1 runs or checks failed, 2 invalid configuration or
usage, 3 an evaluation environment could not be built.
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
EXIT_ENVIRONMENT = 3

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

    p = sub.add_parser(
        "check",
        help="Smoke test an experiment without calling LLMs (unless --llm).",
    )
    _add_config_args(p)
    p.add_argument(
        "--llm",
        action="store_true",
        help="Also send one short prompt to every configured LLM.",
    )
    p.add_argument(
        "--no-methods",
        action="store_true",
        help="Only check the problems, not the methods.",
    )

    p = sub.add_parser("init", help="Create a starter experiment.")
    p.add_argument(
        "directory", nargs="?", default=".", help="Folder to create (default: .)."
    )
    p.add_argument(
        "--template",
        "-t",
        default="builtin",
        choices=["builtin", "custom-problem", "custom-method"],
        help="What to start from (default: builtin).",
    )
    p.add_argument("--force", action="store_true", help="Overwrite existing files.")

    p = sub.add_parser("cache", help="Show or delete cached evaluation environments.")
    p.add_argument("action", choices=["list", "clear"])

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

    p = sub.add_parser(
        "skill",
        help="Install the BLADE skill for coding agents (Claude Code, ...).",
    )
    p.add_argument(
        "directory",
        nargs="?",
        default=".claude/skills",
        help=(
            "Skills folder of your agent; the skill is written to "
            "<directory>/blade (default: .claude/skills)."
        ),
    )
    p.add_argument("--force", action="store_true", help="Overwrite existing files.")
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


def _resume_status(plan) -> list[str]:
    """Lines describing what resuming in ``plan.output_dir`` would do."""
    from .config.record import resume_status

    return resume_status(plan).summary()


def _cmd_run(args) -> int:
    plan = _load_plan(args)
    if args.dry_run:
        print(plan.summary())
        print("\n".join(_resume_status(plan)))
        return EXIT_OK
    from .envs import EnvironmentSetupError

    experiment = plan.make_experiment()
    try:
        experiment()
    except EnvironmentSetupError as e:
        print(f"\n{e}", file=sys.stderr)
        return EXIT_ENVIRONMENT
    except RuntimeError as e:
        print(f"\n{e}", file=sys.stderr)
        return EXIT_RUN_FAILED
    print(f"\nResults: {plan.output_dir}")
    return EXIT_OK


def _cmd_check(args) -> int:
    from .config.check import FAIL, run_checks

    plan = _load_plan(args)
    print(f"Checking {plan.config.experiment.name or args.config} ...\n")
    results = run_checks(
        plan,
        methods=not args.no_methods,
        llms=args.llm,
        report=lambda r: print(r, flush=True),
        verbose=args.verbose,
    )
    failed = sum(r.status == FAIL for r in results)
    counts = {
        s: sum(r.status == s for r in results) for s in ("ok", "warn", "fail", "skip")
    }
    print(
        "\n"
        + ", ".join(f"{n} {s}" for s, n in counts.items() if n)
        + (" - fix the failures before running." if failed else " - ready to run.")
    )
    if not args.llm:
        print("(LLMs were not contacted; add --llm to test them.)")
    return EXIT_RUN_FAILED if failed else EXIT_OK


def _cmd_init(args) -> int:
    from .config.templates import write_template

    try:
        files = write_template(args.template, args.directory, force=args.force)
    except FileExistsError as e:
        print(str(e), file=sys.stderr)
        return EXIT_CONFIG
    print(f"Created a '{args.template}' experiment:")
    for path in files:
        print(f"  {path}")
    config = files[0]
    print(
        f"\nNext: edit the TODOs, then\n"
        f"  uv run iohblade check {config}\n"
        f"  uv run iohblade run {config}"
    )
    return EXIT_OK


def _cmd_cache(args) -> int:
    from .envs import cache_root, clear_cache, list_envs

    if args.action == "clear":
        count = clear_cache()
        print(f"Removed {count} cached environment(s) from {cache_root()}.")
        return EXIT_OK
    envs = list_envs()
    print(f"Cached evaluation environments in {cache_root()}:")
    if not envs:
        print("  (none)")
    total = 0
    for env in envs:
        total += env["size"]
        deps = ", ".join(env.get("dependencies", [])) or "-"
        state = "" if env["complete"] else " [incomplete]"
        print(
            f"  {env['key']}  {env['size'] / 1e6:7.1f} MB  "
            f"last used {env.get('last_used', '?')}{state}\n      {deps}"
        )
    if envs:
        print(f"Total: {total / 1e6:.1f} MB. Remove with: iohblade cache clear")
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
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"Schema written to {args.output}")
    else:
        print(text)
    return EXIT_OK


def _cmd_skill(args) -> int:
    from .agent_skill import install_skill

    try:
        files = install_skill(args.directory, force=args.force)
    except FileExistsError as e:
        print(str(e), file=sys.stderr)
        return EXIT_CONFIG
    print("Installed the BLADE skill:")
    for path in files:
        print(f"  {path}")
    print(
        "\nAsk your coding agent, e.g. 'Set up a BLADE experiment for my "
        "problem'.\nOther agents: pass their skills folder, e.g. "
        "`iohblade skill .agents/skills`."
    )
    return EXIT_OK


_COMMANDS = {
    "check": _cmd_check,
    "init": _cmd_init,
    "cache": _cmd_cache,
    "validate": _cmd_validate,
    "run": _cmd_run,
    "list": _cmd_list,
    "schema": _cmd_schema,
    "skill": _cmd_skill,
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
