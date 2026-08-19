"""Golden dataset loading and offline benchmark scoring.

Per LLD 2.5: "The offline benchmark uses a stronger judge for scoring
accuracy." The offline path in this codebase runs the full
`CompositeEvaluator` (all four sub-evaluators, including `correctness`
against the golden set's `ground_truth`) rather than the latency-bounded
`OnlineJudge`. When a live LLM judge is wired in (see the per-evaluator
docstrings for the upgrade path), swapping the offline judge configuration
happens here, without touching `POST /eval/run` or the online gate.

The shipped golden dataset (`app/evaluation/golden/golden_qa.jsonl`) is a
small, enterprise-contract-flavored regression set of question/context/
answer/ground_truth samples authored for this milestone. It exists so
`pytest tests/unit` and `POST /eval/run` both work with zero external
infrastructure.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.core.interfaces import Evaluator
from app.core.schemas import EvalResult, EvalSample

_DEFAULT_DATASET_PATH = Path(__file__).parent / "golden" / "golden_qa.jsonl"

# Known dataset ids that POST /eval/run may reference, mapped to a file on
# disk. "golden_qa" (also the default) is the dataset shipped with M3.
DATASET_REGISTRY: dict[str, Path] = {
    "golden_qa": _DEFAULT_DATASET_PATH,
    "default": _DEFAULT_DATASET_PATH,
}


def resolve_dataset_path(dataset_id: str | None) -> Path:
    if not dataset_id:
        return _DEFAULT_DATASET_PATH
    if dataset_id in DATASET_REGISTRY:
        return DATASET_REGISTRY[dataset_id]
    candidate = Path(dataset_id)
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"unknown eval dataset_id: {dataset_id!r}")


def load_golden_dataset(path: Path | str | None = None) -> list[EvalSample]:
    """Load a JSONL golden dataset into a list of EvalSample."""
    resolved = Path(path) if path else _DEFAULT_DATASET_PATH
    samples: list[EvalSample] = []
    with resolved.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            samples.append(EvalSample(**row))
    return samples


class SampleScore:
    """Per-sample result paired with the question/answer that produced it,
    convenient for persisting EvalResultRow.
    """

    __slots__ = ("answer", "question", "result")

    def __init__(self, question: str, answer: str, result: EvalResult) -> None:
        self.question = question
        self.answer = answer
        self.result = result


async def run_benchmark(
    evaluator: Evaluator, samples: list[EvalSample]
) -> tuple[list[SampleScore], dict[str, Any]]:
    """Score every sample with `evaluator` and compute an aggregate report.

    Returns (per_sample_scores, aggregate) where aggregate contains mean
    faithfulness/groundedness/relevance/correctness plus sample_count. This
    is what backs both the offline benchmark report and `EvalRun.aggregate`.
    """
    per_sample: list[SampleScore] = []
    for sample in samples:
        result = await evaluator.evaluate(sample)
        per_sample.append(SampleScore(sample.question, sample.answer, result))

    aggregate = aggregate_scores(per_sample)
    return per_sample, aggregate


def aggregate_scores(per_sample: list[SampleScore]) -> dict[str, Any]:
    n = len(per_sample)
    if n == 0:
        return {"sample_count": 0}

    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    faithfulness_vals = [s.result.faithfulness for s in per_sample]
    groundedness_vals = [s.result.groundedness for s in per_sample]
    relevance_vals = [s.result.relevance for s in per_sample]
    correctness_vals = [s.result.correctness for s in per_sample if s.result.correctness is not None]

    return {
        "sample_count": n,
        "faithfulness_mean": _mean(faithfulness_vals),
        "groundedness_mean": _mean(groundedness_vals),
        "relevance_mean": _mean(relevance_vals),
        "correctness_mean": _mean(correctness_vals) if correctness_vals else None,
    }


def passed(result: EvalResult, faithfulness_threshold: float, groundedness_threshold: float) -> bool:
    """Same pass/fail rule the runtime gate uses (LLD 2.4 decision logic:
    groundedness >= tau_g and faithfulness >= tau_f -> finalize), reused
    here to flag each golden-set sample as passed/failed for EvalResultRow.
    """
    return result.faithfulness >= faithfulness_threshold and result.groundedness >= groundedness_threshold
