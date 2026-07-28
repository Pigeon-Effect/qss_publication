"""Command-line entry point.

Both protocols are driven from here so that a run is fully described by its
command line, which the reports then record verbatim::

    python -m clustervalidation intrusion --level h3 --trials 1000
    python -m clustervalidation coherence --level h1 --prompt tan_dsouza
    python -m clustervalidation inspect --level h2
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from clustervalidation import __version__
from clustervalidation.config import (
    DEFAULT_BASE_URL,
    DEFAULT_DB_PATH,
    DEFAULT_MODEL,
    DEFAULT_RESULTS_DIR,
    HIERARCHY_LEVELS,
    MODELS,
    RunConfig,
)
from clustervalidation.corpus import corpus_summary, load_clusters
from clustervalidation.llm import ChatClient, MissingAPIKeyError
from clustervalidation.prompts import (
    COHERENCE_PROMPTS,
    DEFAULT_COHERENCE_PROMPT,
    DEFAULT_INTRUSION_PROMPT,
    INTRUSION_PROMPTS,
)
from clustervalidation.protocols import coherence, intrusion
from clustervalidation.reporting import write_reports

DEFAULT_SEED = 20250628


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--level",
        choices=HIERARCHY_LEVELS,
        required=True,
        help="taxonomy level to validate",
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODELS),
        default=DEFAULT_MODEL,
        help=f"model to query (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=100,
        help="number of trials (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"seed for panel construction (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=200,
        help="truncate each abstract to this many words (default: 200)",
    )
    parser.add_argument(
        "--panel-size",
        type=int,
        default=5,
        help="documents shown per trial (default: 5)",
    )
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="path to the corpus")
    parser.add_argument(
        "--results-dir",
        default=None,
        help="output directory (default: results/<protocol>)",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="API base URL")
    parser.add_argument("--stem", default=None, help="override output filename stem")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="build panels and print the first prompt without calling the API",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="suppress per-trial progress output"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clustervalidation",
        description=(
            "LLM-based validation of the hierarchical AI-research taxonomy."
        ),
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    intrusion_parser = subparsers.add_parser(
        "intrusion", help="run document-intrusion detection"
    )
    _add_common(intrusion_parser)
    intrusion_parser.add_argument(
        "--prompt",
        choices=sorted(INTRUSION_PROMPTS),
        default=DEFAULT_INTRUSION_PROMPT,
        help=f"prompt variant (default: {DEFAULT_INTRUSION_PROMPT})",
    )

    coherence_parser = subparsers.add_parser(
        "coherence", help="run Likert coherence rating"
    )
    _add_common(coherence_parser)
    coherence_parser.add_argument(
        "--prompt",
        choices=sorted(COHERENCE_PROMPTS),
        default=DEFAULT_COHERENCE_PROMPT,
        help=f"prompt variant (default: {DEFAULT_COHERENCE_PROMPT})",
    )

    inspect_parser = subparsers.add_parser(
        "inspect", help="report corpus statistics without querying the API"
    )
    inspect_parser.add_argument("--level", choices=HIERARCHY_LEVELS, required=True)
    inspect_parser.add_argument("--db", default=DEFAULT_DB_PATH)
    inspect_parser.add_argument("--min-cluster-size", type=int, default=5)

    return parser


def _make_config(args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        protocol=args.command,
        level=args.level,
        model=MODELS[args.model],
        prompt_variant=args.prompt,
        trials=args.trials,
        seed=args.seed,
        max_words=args.max_words,
        panel_size=args.panel_size,
        db_path=args.db,
        base_url=args.base_url,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "inspect":
        clusters = load_clusters(args.db, args.level, args.min_cluster_size)
        print(json.dumps(corpus_summary(clusters), indent=2))
        return 0

    config = _make_config(args)
    module = intrusion if args.command == "intrusion" else coherence

    print(f"Loading corpus at level {config.level} ...")
    clusters = load_clusters(config.db_path, config.level, config.min_cluster_size)
    stats = corpus_summary(clusters)
    print(
        f"  {stats['clusters']} clusters, {stats['documents']:,} documents "
        f"(smallest {stats['smallest_cluster']}, largest {stats['largest_cluster']})"
    )

    if args.dry_run:
        return _dry_run(module, clusters, config)

    try:
        client = ChatClient(config.model, base_url=config.base_url)
    except MissingAPIKeyError as error:
        print(str(error), file=sys.stderr)
        return 2

    print(
        f"\nRunning {config.protocol} | level={config.level} "
        f"| model={config.model.name} | prompt={config.prompt_variant} "
        f"| trials={config.trials} | seed={config.seed}\n"
    )
    outcome = module.run(clusters, config, client, progress=not args.quiet)

    print(f"\n{'=' * 60}")
    for key, value in outcome.summary().items():
        print(f"{key:>28}: {value}")
    print("=" * 60)

    results_dir = args.results_dir or os.path.join(DEFAULT_RESULTS_DIR, args.command)
    paths = write_reports(outcome, results_dir, args.stem)
    print("\nReports written:")
    for extension, path in paths.items():
        print(f"  {extension:>5}  {path}")
    return 0


def _dry_run(module, clusters, config: RunConfig) -> int:
    """Show what would be sent, without spending API credit."""
    if config.protocol == "intrusion":
        panels = module.build_panels(clusters, config)
        first = panels[0]
        from clustervalidation.prompts import INTRUSION_PROMPTS, render_panel

        rendered = render_panel(
            [(item.document.title, item.document.abstract) for item in first.items]
        )
        prompt = INTRUSION_PROMPTS[config.prompt_variant](rendered, config.panel_size)
        print(f"\nBuilt {len(panels)} panels. First panel:")
        print(f"  home cluster     {first.home_cluster}")
        print(f"  intruder cluster {first.intruder_cluster}")
        print(f"  true position    {first.true_position}")
    else:
        samples = module.build_samples(clusters, config)
        first = samples[0]
        from clustervalidation.prompts import COHERENCE_PROMPTS, render_panel

        rendered = render_panel([(d.title, d.abstract) for d in first.documents])
        prompt = COHERENCE_PROMPTS[config.prompt_variant](
            rendered, config.panel_size, config.level
        )
        print(f"\nBuilt {len(samples)} samples. First sample:")
        print(f"  cluster {first.cluster}")

    print(f"\n{'-' * 60}\n{prompt}\n{'-' * 60}")
    print("\nDry run: no API calls were made.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
