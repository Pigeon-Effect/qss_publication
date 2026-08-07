"""Writing run results to disk.

Each run produces three files sharing a stem:

``<stem>.json``
    Run manifest and summary statistics. This is the file to read when
    aggregating across runs or rebuilding a table for the manuscript.
``<stem>.jsonl``
    One record per trial, for re-analysis without re-querying the API.
``<stem>.txt``
    Human-readable transcript, in the same layout as the archived runs in
    ``results/`` so old and new output can be read side by side.

Every file carries the run manifest, so a result is never separated from the
parameters that produced it.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from clustervalidation.config import LEVEL_DESCRIPTIONS, RunConfig
from clustervalidation.protocols import coherence, intrusion

_RULE = "=" * 70
_SEPARATOR = "-" * 70


def default_stem(config: RunConfig) -> str:
    """Build a descriptive, sortable filename stem for a run."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model = config.model.name.replace(".", "-")
    return (
        f"{config.protocol}_{config.level}_{model}_"
        f"{config.prompt_variant}_n{config.trials}_seed{config.seed}_{stamp}"
    )


def _manifest(config: RunConfig, summary: dict) -> dict:
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "configuration": config.as_dict(),
        "summary": summary,
    }


def write_reports(
    outcome: intrusion.Outcome | coherence.Outcome,
    results_dir: str,
    stem: str | None = None,
) -> dict[str, str]:
    """Write all three report files and return their paths by extension."""
    config = outcome.config
    stem = stem or default_stem(config)
    os.makedirs(results_dir, exist_ok=True)

    summary = outcome.summary()
    manifest = _manifest(config, summary)

    paths = {
        "json": os.path.join(results_dir, f"{stem}.json"),
        "jsonl": os.path.join(results_dir, f"{stem}.jsonl"),
        "txt": os.path.join(results_dir, f"{stem}.txt"),
    }

    with open(paths["json"], "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    is_intrusion = isinstance(outcome, intrusion.Outcome)
    records = (
        _intrusion_records(outcome) if is_intrusion else _coherence_records(outcome)
    )
    with open(paths["jsonl"], "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    with open(paths["txt"], "w", encoding="utf-8") as handle:
        if is_intrusion:
            _write_intrusion_text(handle, outcome, manifest)
        else:
            _write_coherence_text(handle, outcome, manifest)

    return paths


# --------------------------------------------------------------------------
# JSONL records
# --------------------------------------------------------------------------


def _intrusion_records(outcome: intrusion.Outcome) -> list[dict[str, Any]]:
    """Build one self-describing record per trial."""
    return [intrusion_record(outcome.config, trial) for trial in outcome.trials]


def intrusion_record(config: RunConfig, trial: intrusion.Trial) -> dict[str, Any]:
    """Build one self-describing record for a single trial.

    Each record repeats the run identifiers (level, model, prompt, seed) rather
    than relying on the manifest alone, so a single line remains interpretable
    when records are concatenated across runs or shared in isolation. The
    abstract stored is the truncated text actually sent to the model, not the
    full source abstract, so the prompt is reconstructable from the record.

    Exposed separately from the batch writer so a long run can checkpoint each
    trial as it completes, rather than holding every result in memory until the
    end and losing all of it if the process dies.
    """
    return {
        "trial": trial.index,
        "timestamp_utc": trial.timestamp_utc,
        "level": config.level,
        "level_description": LEVEL_DESCRIPTIONS[config.level],
        "model": config.model.name,
        "thinking": config.model.thinking,
        "prompt_variant": config.prompt_variant,
        "seed": config.seed,
        "panel_size": config.panel_size,
        "max_words": config.max_words,
        "home_cluster": trial.panel.home_cluster,
        "intruder_cluster": trial.panel.intruder_cluster,
        "true_position": trial.panel.true_position,
        "predicted": trial.predicted,
        "correct": trial.correct,
        "extraction_rule": trial.extraction_rule,
        "recovered_from_reasoning": trial.recovered_from_reasoning,
        "forced_choice": trial.forced_choice,
        "forced_guess": trial.forced_guess,
        "finish_reason": trial.finish_reason,
        "truncated": trial.truncated,
        "cost_usd": round(trial.cost_usd, 8),
        "error": trial.error,
        "panel": [
            {
                "position": position,
                "document_id": item.document.id,
                "title": item.document.title,
                "abstract": item.document.abstract,
                "is_intruder": item.is_intruder,
            }
            for position, item in enumerate(trial.panel.items, start=1)
        ],
        "response": trial.content,
        "reasoning": trial.reasoning,
    }


def _coherence_records(outcome: coherence.Outcome) -> list[dict[str, Any]]:
    records = []
    for trial in outcome.trials:
        records.append(
            {
                "trial": trial.index,
                "cluster": trial.sample.cluster,
                "rating": trial.rating,
                "topic_rating": trial.topic_rating,
                "method_rating": trial.method_rating,
                "extraction_rule": trial.extraction_rule,
                "finish_reason": trial.finish_reason,
                "truncated": trial.truncated,
                "cost_usd": round(trial.cost_usd, 8),
                "error": trial.error,
                "documents": [
                    {"document_id": d.id, "title": d.title}
                    for d in trial.sample.documents
                ],
                "response": trial.content,
                "reasoning": trial.reasoning,
            }
        )
    return records


# --------------------------------------------------------------------------
# Human-readable transcripts
# --------------------------------------------------------------------------


def _write_header(handle, title: str, manifest: dict) -> None:
    config = manifest["configuration"]
    handle.write(f"{title}\n")
    handle.write(f"Generated: {manifest['generated_utc']}\n\n")
    handle.write("CONFIGURATION\n")
    for key, value in config.items():
        handle.write(f"  {key}: {value}\n")
    handle.write("\nSUMMARY\n")
    for key, value in manifest["summary"].items():
        handle.write(f"  {key}: {value}\n")
    handle.write(f"\n{_RULE}\n\n")


def _write_intrusion_text(handle, outcome: intrusion.Outcome, manifest: dict) -> None:
    _write_header(handle, "DOCUMENT INTRUSION DETECTION", manifest)

    for trial in outcome.trials:
        handle.write(f"Trial {trial.index}\n")
        handle.write(f"Timestamp: {trial.timestamp_utc}\n")
        handle.write(f"Home cluster: {trial.panel.home_cluster}\n")
        handle.write(f"Intruder cluster: {trial.panel.intruder_cluster}\n")
        handle.write(f"True intruder position: {trial.panel.true_position}\n")
        handle.write(f"Predicted: {trial.predicted}\n")
        handle.write(f"Correct: {trial.correct}\n")
        handle.write(f"Extraction rule: {trial.extraction_rule}\n")
        handle.write(f"Recovered from reasoning: {trial.recovered_from_reasoning}\n")
        handle.write(f"Finish reason: {trial.finish_reason}\n")
        handle.write(f"Cost: ${trial.cost_usd:.6f}\n")
        if trial.error:
            handle.write(f"ERROR: {trial.error}\n")
        if trial.content:
            handle.write(f"\nResponse:\n{trial.content}\n")
        if trial.reasoning:
            handle.write(f"\nReasoning trace:\n{trial.reasoning}\n")

        handle.write("\nPanel:\n")
        for position, item in enumerate(trial.panel.items, start=1):
            marker = "  *** INTRUDER ***" if item.is_intruder else ""
            handle.write(f"  [{position}] ID={item.document.id}{marker}\n")
            handle.write(f"      Title: {item.document.title}\n")
            handle.write(f"      Abstract: {item.document.abstract[:500]}\n\n")
        handle.write(f"{_SEPARATOR}\n\n")


def _write_coherence_text(handle, outcome: coherence.Outcome, manifest: dict) -> None:
    _write_header(handle, "CLUSTER COHERENCE RATING", manifest)

    for trial in outcome.trials:
        handle.write(f"Trial {trial.index}\n")
        handle.write(f"Cluster: {trial.sample.cluster}\n")
        handle.write(f"Rating: {trial.rating}\n")
        if trial.topic_rating is not None or trial.method_rating is not None:
            handle.write(f"Topic coherence: {trial.topic_rating}\n")
            handle.write(f"Methodology coherence: {trial.method_rating}\n")
        handle.write(f"Extraction rule: {trial.extraction_rule}\n")
        handle.write(f"Cost: ${trial.cost_usd:.6f}\n")
        if trial.error:
            handle.write(f"ERROR: {trial.error}\n")
        if trial.content:
            handle.write(f"\nResponse:\n{trial.content}\n")

        handle.write("\nDocuments:\n")
        for position, document in enumerate(trial.sample.documents, start=1):
            handle.write(f"  [{position}] ID={document.id}\n")
            handle.write(f"      Title: {document.title}\n")
            handle.write(f"      Abstract: {document.abstract[:500]}\n\n")
        handle.write(f"{_SEPARATOR}\n\n")
