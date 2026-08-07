"""Static configuration: taxonomy levels, model specifications and pricing.

Nothing here reads the environment or the filesystem; it is pure data, so the
values that determine an experiment can be inspected and cited directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Repository layout
# --------------------------------------------------------------------------
# Package lives at <root>/src/clustervalidation/config.py, so the root is
# three levels up. Everything else is derived from it, which keeps the code
# independent of where the repository is checked out.
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
DEFAULT_DB_PATH = os.path.join(PROJECT_ROOT, "data", "merged_works_labeled.db")
DEFAULT_RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# --------------------------------------------------------------------------
# Taxonomy hierarchy
# --------------------------------------------------------------------------
HIERARCHY_LEVELS = ("h1", "h2", "h3")

LEVEL_DESCRIPTIONS = {
    "h1": "macro level (domains)",
    "h2": "meso level (fields)",
    "h3": "micro level (research fronts)",
}

# A cluster is identified by the concatenation of its labels down to the
# requested level, so that e.g. h2 cluster "34" is field 4 within domain 3.
CLUSTER_ID_SQL = {
    "h1": "CAST(h1_cluster AS TEXT)",
    "h2": "CAST(h1_cluster AS TEXT) || CAST(h2_cluster AS TEXT)",
    "h3": (
        "CAST(h1_cluster AS TEXT) || CAST(h2_cluster AS TEXT) "
        "|| CAST(h3_cluster AS TEXT)"
    ),
}

# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
DEFAULT_BASE_URL = "https://api.deepseek.com"


@dataclass(frozen=True)
class ModelSpec:
    """A model and the decoding settings used to query it.

    ``prompt_usd_per_1m`` and ``completion_usd_per_1m`` are list prices at the
    time the reported experiments were run (June 2025). They are recorded so
    that reported run costs remain reconstructable; they are not fetched live
    and will drift from current pricing.
    """

    name: str
    thinking: bool = False
    temperature: float | None = 0.0
    max_tokens: int = 3000
    timeout: int = 60
    prompt_usd_per_1m: float = 0.14
    completion_usd_per_1m: float = 0.28

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Return the USD cost of one completion."""
        return (
            prompt_tokens * self.prompt_usd_per_1m
            + completion_tokens * self.completion_usd_per_1m
        ) / 1_000_000


# Models used across the reported and exploratory runs. `deepseek-v4-flash`
# with reasoning enabled is the configuration reported in the manuscript.
MODELS: dict[str, ModelSpec] = {
    "deepseek-v4-flash": ModelSpec(
        name="deepseek-v4-flash",
        thinking=True,
        temperature=None,  # reasoning models ignore temperature
        max_tokens=3000,
    ),
    "deepseek-chat": ModelSpec(
        name="deepseek-chat",
        thinking=False,
        temperature=0.0,
        max_tokens=200,
    ),
    "deepseek-reasoner": ModelSpec(
        name="deepseek-reasoner",
        thinking=True,
        temperature=None,
        max_tokens=3000,
    ),
    "deepseek-v4-pro": ModelSpec(
        name="deepseek-v4-pro",
        thinking=True,
        temperature=None,
        max_tokens=3000,
    ),
}

DEFAULT_MODEL = "deepseek-v4-flash"


# --------------------------------------------------------------------------
# Experiment configuration
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class RunConfig:
    """Everything needed to reproduce a single experimental run.

    This object is serialised verbatim into the run manifest, so a result file
    always carries the parameters that produced it.
    """

    protocol: str
    level: str
    model: ModelSpec
    prompt_variant: str
    trials: int = 100
    seed: int = 20250628
    max_words: int = 200
    panel_size: int = 5
    min_cluster_size: int = 5
    db_path: str = DEFAULT_DB_PATH
    base_url: str = DEFAULT_BASE_URL
    # Cheap non-reasoning model used to obtain a verdict when the primary
    # response gave none. `None` disables the follow-up, leaving an unanswered
    # trial to the seeded guess.
    force_choice_model: ModelSpec | None = None
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        """Return a JSON-serialisable view for the run manifest."""
        return {
            "protocol": self.protocol,
            "level": self.level,
            "level_description": LEVEL_DESCRIPTIONS[self.level],
            "model": self.model.name,
            "thinking": self.model.thinking,
            "temperature": self.model.temperature,
            "max_tokens": self.model.max_tokens,
            "prompt_variant": self.prompt_variant,
            "trials": self.trials,
            "seed": self.seed,
            "max_words": self.max_words,
            "panel_size": self.panel_size,
            "min_cluster_size": self.min_cluster_size,
            "force_choice_model": (
                self.force_choice_model.name if self.force_choice_model else None
            ),
            "package_version": _version(),
            **self.extra,
        }


def _version() -> str:
    from clustervalidation import __version__

    return __version__
