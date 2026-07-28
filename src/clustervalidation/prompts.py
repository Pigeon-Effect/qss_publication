"""Named prompt templates for both validation protocols.

Every variant that was run during development is kept here verbatim, because
the wording is an experimental parameter: the exploratory results in
``results/exploratory/`` differ from each other only by which template was used.
Templates are addressed by name from the command line (``--prompt``) and the
name is recorded in every run manifest.

Adding a variant is fine; editing an existing one silently invalidates the
results that cite it. Register a new name instead.
"""

from __future__ import annotations

from typing import Callable

# --------------------------------------------------------------------------
# Panel rendering
# --------------------------------------------------------------------------


def render_panel(items: list[tuple[str, str]], with_titles: bool = True) -> str:
    """Render ``(title, abstract)`` pairs as a numbered block.

    Numbering is 1-based and matches the verdict the model is asked to return.
    """
    blocks = []
    for index, (title, abstract) in enumerate(items, start=1):
        if with_titles:
            blocks.append(f"[{index}] Title: {title}\nAbstract: {abstract}")
        else:
            blocks.append(f"[{index}] {abstract}")
    return "\n\n".join(blocks)


# --------------------------------------------------------------------------
# Intrusion-detection prompts
# --------------------------------------------------------------------------

def _intrusion_reasoned(panel: str, panel_size: int) -> str:
    """Reported variant: brief reasoning, then a fixed verdict line."""
    return (
        f"Below are {panel_size} academic papers. "
        "Four share the same research topic; one does not.\n\n"
        f"{panel}\n\n"
        "Reason briefly, then end with a line exactly:\n"
        f"Final verdict: <single digit 1-{panel_size}>"
    )


def _intrusion_minimal(panel: str, panel_size: int) -> str:
    """Digit-only answer, no reasoning. Cheapest variant."""
    return (
        "You are a research librarian. Below are "
        f"{panel_size} paper abstracts (each with title).\n"
        "Four of them belong to the same narrow scientific subfield. "
        "One abstract is from a different subfield and does not belong.\n"
        "Identify the intruder abstract.\n"
        f"Answer with only the number (1-{panel_size}) of the intruder.\n\n"
        f"{panel}"
    )


def _intrusion_expert(panel: str, panel_size: int) -> str:
    """Early variant framing the model as an AI-research expert."""
    return (
        "You are an expert in AI research. I will show you "
        f"{panel_size} paper abstracts. "
        "4 of them belong to the exact same AI subfield. "
        "1 abstract does not belong. "
        'Identify the intruder abstract. Answer ONLY with the number of the '
        'intruder (e.g., "3").\n\n'
        f"{panel}"
    )


def _intrusion_narrow(panel: str, panel_size: int) -> str:
    """Variant stressing highly specific subfields, used at the h3 level."""
    return (
        "You are a research librarian with expertise in many narrow "
        "scientific subfields.\n"
        f"Below are {panel_size} paper abstracts. 4 of them belong to the "
        "exact same highly specific research subfield. "
        "1 abstract is from a different subfield and does not belong.\n"
        "Identify the intruder abstract.\n"
        f"Answer with only the number (1-{panel_size}) of the intruder.\n\n"
        f"{panel}"
    )


def _intrusion_chain_of_thought(panel: str, panel_size: int) -> str:
    """Variant asking the model to name the shared subfield before deciding."""
    return (
        f"You are a research librarian. Below are {panel_size} paper abstracts.\n"
        "Some of them belong to the same narrow scientific subfield. "
        "One is from a completely different discipline.\n"
        "First, say what the common subfield of the others is (one sentence). "
        "Then write 'Intruder: X' where X is the number of the abstract that "
        "does not belong.\n\n"
        f"{panel}"
    )


def _intrusion_structured(panel: str, panel_size: int) -> str:
    """Variant eliciting the inferred home and intruder topics alongside the verdict."""
    return (
        f"You are a research expert. Below are {panel_size} paper abstracts "
        "with titles.\n"
        "Four belong to the SAME research subfield. One is from a DIFFERENT "
        "subfield.\n\n"
        f"{panel}\n\n"
        "Analyze briefly and then output EXACTLY the following:\n"
        "Home cluster topic: <one sentence>\n"
        "Intruder topic: <one sentence>\n"
        f"Final verdict: <single digit 1-{panel_size}>\n"
        "Do not add any extra commentary."
    )


def _intrusion_topic_only(panel: str, panel_size: int) -> str:
    """Topic half of the topic/methodology separation experiment."""
    return (
        "You are a research librarian. Below are "
        f"{panel_size} paper abstracts (each with title).\n"
        "Four of them belong to the same narrow research topic. "
        "One abstract is from a different topic and does not belong.\n"
        "Identify the intruder abstract based on its research topic.\n"
        f"Answer with only the number (1-{panel_size}) of the intruder.\n\n"
        f"{panel}"
    )


def _intrusion_method_only(panel: str, panel_size: int) -> str:
    """Methodology half of the topic/methodology separation experiment."""
    return (
        "You are a research librarian. Below are "
        f"{panel_size} paper abstracts (each with title).\n"
        "Four of them use the same or very similar research methodology. "
        "One abstract uses a clearly different methodology and does not "
        "belong.\n"
        "Identify the intruder abstract based on its methodology.\n"
        f"Answer with only the number (1-{panel_size}) of the intruder.\n\n"
        f"{panel}"
    )


IntrusionPrompt = Callable[[str, int], str]

INTRUSION_PROMPTS: dict[str, IntrusionPrompt] = {
    "reasoned": _intrusion_reasoned,
    "minimal": _intrusion_minimal,
    "expert": _intrusion_expert,
    "narrow": _intrusion_narrow,
    "chain_of_thought": _intrusion_chain_of_thought,
    "structured": _intrusion_structured,
    "topic_only": _intrusion_topic_only,
    "method_only": _intrusion_method_only,
}

DEFAULT_INTRUSION_PROMPT = "reasoned"


# --------------------------------------------------------------------------
# Coherence-rating prompts
# --------------------------------------------------------------------------

COHERENCE_SYSTEM_MESSAGE = (
    "You are an expert research evaluator. "
    "Your task is to assess the coherence of a group of research paper "
    "abstracts, each with its title."
)

# The Tan & D'Souza (2025) template asks about a different unit of analysis at
# each hierarchy level, matching what that level is meant to capture.
_TAN_DSOUZA_UNITS = {
    "h1": (
        "single, recognizable **broad scientific discipline** (e.g., "
        "'Computer Science', 'Medicine', 'Physics', 'Biology', etc.)",
        "discipline-level group",
        "no shared broad discipline",
        "all clearly belong to the same broad discipline",
    ),
    "h2": (
        "single, recognizable **research subfield or closely related area** "
        "(e.g., 'Deep Learning', 'Cancer Biology', 'Quantum Computing', "
        "'Medieval Literature', etc.)",
        "subfield-level group",
        "no shared subfield",
        "all clearly belong to the same subfield",
    ),
    "h3": (
        "single, recognizable **specific research topic** (e.g., "
        "'Adversarial Robustness', 'Retinal Image Segmentation', "
        "'Federated Learning Privacy', etc.)",
        "topic-level group",
        "no shared research topic",
        "all clearly address the same specific research topic",
    ),
}


def _coherence_tan_dsouza(panel: str, panel_size: int, level: str) -> str:
    """Likert coherence rating, after Tan and D'Souza (2025)."""
    unit, group, low, high = _TAN_DSOUZA_UNITS[level]
    return (
        f"You will be given a set of {panel_size} abstracts that an "
        "algorithmic model has grouped together.\n\n"
        f"For this set, determine if the {panel_size} abstracts collectively "
        f"belong to a {unit}.\n\n"
        "Provide your answer in two parts:\n"
        "1. First, provide a 1-sentence explanation of why the abstracts do "
        f"or do not form a coherent {group}.\n"
        "2. Then, state your coherence rating on a 5-point Likert scale, "
        "where:\n"
        f"   - 1 = Completely incoherent ({low})\n"
        "   - 2 = Somewhat incoherent\n"
        "   - 3 = Moderately coherent\n"
        "   - 4 = Very coherent\n"
        f"   - 5 = Highly coherent ({high})\n\n"
        "Provide the final rating exactly in the format: `Rating: X`.\n\n"
        "---\n"
        "Abstracts to Evaluate:\n"
        f"{panel}"
    )


def _coherence_minimal(panel: str, panel_size: int, level: str) -> str:
    """Single-digit rating with no explanation."""
    return (
        f"You are a research librarian. Below are {panel_size} paper abstracts "
        "that have been algorithmically grouped together.\n"
        "Rate how coherent this collection is - do they all belong to a single "
        "recognizable scientific subfield or closely related area?\n"
        "Use a scale from 1 (completely unrelated) to 5 (very coherent, "
        "clearly one subfield).\n"
        "Answer with only the number (1-5).\n\n"
        f"{panel}"
    )


def _coherence_dual_score(panel: str, panel_size: int, level: str) -> str:
    """Rates topical and methodological coherence separately.

    Motivated by the observation that a cluster can be methodologically tight
    while topically diffuse (or the reverse), which a single score conflates.
    """
    return (
        f"You will be given a set of {panel_size} abstracts that an algorithm "
        "grouped together.\n\n"
        "First, reason step by step:\n\n"
        "Step 1 - Read each abstract and identify the main research topic "
        "(what they study) and the main methodology (how they study it).\n\n"
        "Step 2 - For Topic Coherence, evaluate how well these abstracts "
        "belong together as a single research subfield. Use this scale:\n\n"
        "1 = No coherent topic - abstracts are from completely different "
        "major areas.\n"
        "2 = Weak topic relation - they share only a very broad area like "
        "artificial intelligence but no clear subfield.\n"
        "3 = Moderate topic coherence - they belong to the same general "
        "subfield, but within that subfield they cover diverse specific "
        "problems.\n"
        "4 = Strong topic coherence - they consistently focus on a narrower "
        "theme within a subfield.\n"
        "5 = Perfect topic coherence - all abstracts clearly address the same "
        "specific research topic.\n\n"
        "Step 3 - For Methodology Coherence, evaluate how similar their "
        "research methods are. Use this scale:\n\n"
        "1 = No shared methodology - methods are completely different.\n"
        "2 = Weak method overlap - only at the highest abstraction level.\n"
        "3 = Moderate method coherence - they share a common family of "
        "methods, but with meaningful differences.\n"
        "4 = Strong method coherence - they use the same specific methodology "
        "throughout.\n"
        "5 = Identical methodology - the experimental setups, models, and "
        "evaluation procedures are essentially the same.\n\n"
        "Finally, output exactly these two lines and nothing else:\n"
        "Topic coherence: <1-5>\n"
        "Methodology coherence: <1-5>\n\n"
        "---\n"
        "Abstracts to Evaluate:\n"
        f"{panel}"
    )


CoherencePrompt = Callable[[str, int, str], str]

COHERENCE_PROMPTS: dict[str, CoherencePrompt] = {
    "tan_dsouza": _coherence_tan_dsouza,
    "minimal": _coherence_minimal,
    "dual_score": _coherence_dual_score,
}

DEFAULT_COHERENCE_PROMPT = "tan_dsouza"
