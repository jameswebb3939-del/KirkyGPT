from __future__ import annotations

from llm_followups.eval.evaluators.rubric import Rubric


COHERENCE_RUBRIC = Rubric(
    name="coherence",
    description="Evaluate how coherent, well-structured, and easy to follow the response is.",
    score_levels={
        0: "Incoherent or unreadable",
        1: "Weakly coherent",
        2: "Mostly coherent",
        3: "Very coherent, well-structured, and easy to follow",
    },
    pass_threshold=2.0,
)


RELEVANCE_RUBRIC = Rubric(
    name="answer_relevance",
    description="Evaluate how directly the response addresses the original input and how useful it is.",
    score_levels={
        0: "Not relevant to the input",
        1: "Slightly relevant",
        2: "Mostly relevant",
        3: "Directly relevant and useful",
    },
    pass_threshold=2.0,
)
