import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import(
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness
)
from pydantic import BaseModel, Field

from Agents.rag_agent import answer, retrieve_context
from config.settings import settings

logger = logging.getLogger(__name__)


class EvalSample(BaseModel):
    question: str
    ground_truth: str = ""


class EvalResult(BaseModel):
    question: str
    answer: str
    faithfulness: float = Field(default=0.0)
    answer_relevancy: float = Field(default=0.0)
    context_precision: float = Field(default=0.0)
    context_recall: float = Field(default=0.0)
    answer_correctness: float = Field(default=0.0)
    has_ground_truth: bool = False
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class EvalReport(BaseModel):
    total_samples: int
    avg_faithfulness: float
    avg_answer_relevancy: float
    avg_context_precision: float
    avg_context_recall: float
    avg_answer_correctness: float
    results: List[EvalResult]
    generated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


def evaluate_single(sample: EvalSample) -> EvalResult:
    rag_response = answer(sample.question)
    docs = retrieve_context(sample.question)
    contexts = [d.page_content for d in docs]
    row = {
        "question": [sample.question],
        "answer": [rag_response.answer],
        "contexts": [contexts]
    }
    if sample.ground_truth:
        row["ground_truth"] = [sample.ground_truth]
    dataset = Dataset.from_dict(row)
    metrics = [faithfulness, answer_relevancy, context_precision]
    if sample.ground_truth:
        metrics += [context_recall, answer_correctness]
    try:
        result = evaluate(dataset, metrics=metrics)
        df = result.to_pandas()
        row_data = df.iloc[0]
        return EvalResult(
            question=sample.question,
            answer=rag_response.answer,
            faithfulness=float(row_data.get("faithfulness", 0)),
            answer_relevancy=float(row_data.get("answer_relevancy", 0)),
            context_precision=float(row_data.get("context_precision", 0)),
            context_recall=float(row_data.get("context_recall", 0)) if sample.ground_truth else 0.0,
            answer_correctness=float(row_data.get("answer_correctness", 0)) if sample.ground_truth else 0.0,
            has_ground_truth=bool(sample.ground_truth)
        )
    except Exception as e:
        logger.error(f"RAGAS Evaluation failed: {e}")
        return EvalResult(
            question=sample.question,
            answer=rag_response.answer,
            has_ground_truth=bool(sample.ground_truth),
        )


def evaluate_batch(samples: List[EvalSample],
                   output_path: str | None = None) -> EvalReport:
    logger.info(f"Starting batch RAGAS evaluation on {len(samples)} samples")
    questions, answers_list, contexts_list, ground_truths = [], [], [], []
    for s in samples:
        rag_resp = answer(s.question)
        docs = retrieve_context(s.question)
        questions.append(s.question)
        answers_list.append(rag_resp.answer)
        contexts_list.append([d.page_content for d in docs])
        ground_truths.append(s.ground_truth)
    has_gt = any(ground_truths)
    row = {
        "question": questions,
        "answer": answers_list,
        "contexts": contexts_list
    }
    if has_gt:
        row["ground_truth"] = ground_truths
    dataset = Dataset.from_dict(row)
    metrics = [faithfulness, answer_relevancy, context_precision]
    if has_gt:
        metrics += [context_recall, answer_correctness]

    try:
        result = evaluate(dataset, metrics=metrics)
        df = result.to_pandas()
    except Exception as e:
        logger.error(f"Batch RAGAS evaluation failed: {e}")
        df = None

    eval_results = []
    for i, s in enumerate(samples):
        if df is not None:
            row_data = df.iloc[i]
            er = EvalResult(
                question=s.question,
                answer=answers_list[i],
                faithfulness=float(row_data.get("faithfulness", 0)),
                answer_relevancy=float(row_data.get("answer_relevancy", 0)),
                context_precision=float(row_data.get("context_precision", 0)),
                context_recall=float(row_data.get("context_recall", 0)) if has_gt else 0.0,
                answer_correctness=float(row_data.get("answer_correctness", 0)) if has_gt else 0.0,
                has_ground_truth=bool(s.ground_truth),
            )
        else:
            er = EvalResult(question=s.question, answer=answers_list[i])
        eval_results.append(er)

    def avg(field_name: str) -> float:
        vals = [getattr(r, field_name) for r in eval_results]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    report = EvalReport(
        total_samples=len(eval_results),
        avg_faithfulness=avg("faithfulness"),
        avg_answer_relevancy=avg("answer_relevancy"),
        avg_context_precision=avg("context_precision"),
        avg_context_recall=avg("context_recall"),
        avg_answer_correctness=avg("answer_correctness"),
        results=eval_results,
    )

    # Persist
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report.model_dump(), f, indent=2)
        logger.info(f"Evaluation report saved to {output_path}")

    logger.info(
        f"Evaluation complete — "
        f"faithfulness={report.avg_faithfulness:.3f} | "
        f"relevancy={report.avg_answer_relevancy:.3f} | "
        f"precision={report.avg_context_precision:.3f}"
    )

    return report


def run_eval_from_file(json_path: str, output_path: str = "data/eval_results.json") -> EvalReport:
    with open(json_path) as f:
        raw = json.load(f)
    samples = [EvalSample(**item) for item in raw]
    return evaluate_batch(samples, output_path=output_path)
