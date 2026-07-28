"""LLM-based validation of a hierarchical topic taxonomy of AI research.

This package implements the two validation protocols reported in Pfundstein,
Efer and Burghardt, *Mapping the AI Research of China, the US and the EU*:

``intrusion``
    Document-intrusion detection adapted from the "reading tea leaves"
    paradigm of Chang et al. (2009). A panel of genuine cluster members is
    shown to a language model together with one intruder drawn from a
    different cluster at the same hierarchy level; detection accuracy is the
    share of correctly identified intruders.

``coherence``
    Direct Likert-scale coherence rating of a cluster sample, following the
    prompt design of Tan and D'Souza (2025).

Both protocols run against the labelled OpenAlex corpus described in
``data/README.md`` and are driven from the command line::

    python -m clustervalidation intrusion --level h3 --trials 100
    python -m clustervalidation coherence --level h1 --trials 50
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
