import hashlib 
import json 
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime 
from pathlib import Path
from typing import List, Dict, Any, Callable
from functools import wraps

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

def track_latency(name: str):
    def decorator(fn: Callable):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = fn(*args, **kwargs)
            elapsed = time.perf_counter() - start
            MetricsStore.record_latency(name, elapsed)
            return result
        return wrapper
    return decorator


@dataclass
class RequestMetrics:
    query: str
    retrieval_mode: str
    vector_doc_count: int = 0
    graph_doc_count: int = 0
    total_docs_retrieved: int = 0
    cache_hit: bool = False
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    confidence: float = 0.0
    has_sufficient_context: bool = True
    source_diversity: float = 0.0
    hallucination_risk: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


def compute_source_diversity(docs: List[Document]) -> float:
    if not docs:
        return 0.0
    sources = {d.metadata.get("source", "unknown") for d in docs}
    return round(len(sources) / len(docs), 4)


def compute_context_compression_ratio(docs: List[Document], answer: str) -> float:
    total_ctx = sum(len(d.page_content) for d in docs)
    if total_ctx == 0:
        return 0.0
    return round(len(answer) / total_ctx, 4)


def compute_hallucination_risk(answer: str, docs: List[Document]) -> float:
    if not docs or not answer:
        return 1.0
    context_text = " ".join(d.page_content.lower() for d in docs)
    answer_words = set(answer.lower().split())
    context_words = set(context_text.split())
    overlap = answer_words & context_words
    if not answer_words:
        return 0.0
    return round(1.0 - len(overlap) / len(answer_words), 4)


class _MetricsStore:
    def __init__(self):
        self._latencies: Dict[str, list[float]] = defaultdict(list)
        self._requests: List[RequestMetrics] = []
        self._cache_hits = 0
        self._cache_misses = 0

    def record_latency(self, name: str, seconds: float):
        self._latencies[name].append(seconds * 1000)

    def record_request(self, metrics: RequestMetrics):
        self._requests.append(metrics)
        if metrics.cache_hit:
            self._cache_hits += 1
        else:
            self._cache_misses += 1

    def get_summary(self) -> Dict[str, Any]:
        summary = {
            "total_requests": len(self._requests),
            "cache_hit_rate": (
                self._cache_hits / (self._cache_hits + self._cache_misses)
                if (self._cache_hits + self._cache_misses) > 0
                else 0.0
            ),
            "latencies_ms": {},
        }
        for name, lats in self._latencies.items():
            if lats:
                summary["latencies_ms"][name] = {
                    "p50": sorted(lats)[len(lats) // 2],
                    "p95": sorted(lats)[int(len(lats) * 0.95)],
                    "p99": sorted(lats)[int(len(lats) * 0.99)],
                    "avg": sum(lats) / len(lats),
                    "count": len(lats)
                }
        if self._requests:
            avg_conf = sum(r.confidence for r in self._requests) / len(self._requests)
            avg_div = sum(r.source_diversity for r in self._requests) / len(self._requests)
            avg_hall = sum(r.hallucination_risk for r in self._requests) / len(self._requests)
            retrieval_modes = defaultdict(int)
            for r in self._requests:
                retrieval_modes[r.retrieval_mode] += 1
            summary.update({
                "avg_confidence": round(avg_conf, 4),
                "avg_source_diversity": round(avg_div, 4),
                "avg_hallucination_risk": round(avg_hall, 4),
                "retrieval_mode_breakdown": dict(retrieval_modes),
            })

        return summary

    def save(self, path: str = "data/metrics.json"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary": self.get_summary(),
            "requests": [asdict(r) for r in self._requests[-100:]]
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"Metrics saved to {path}")

    def reset(self):
        self._latencies.clear()
        self._requests.clear()
        self._cache_hits = 0
        self._cache_misses = 0


MetricsStore = _MetricsStore()


def export_prometheus_metrics() -> str:
    summary = MetricsStore.get_summary()
    lines = [
        f"# HELP genai_total_requests Total number of RAG requests",
        f"# TYPE genai_total_requests counter",
        f"genai_total_requests {summary['total_requests']}",
        f"",
        f"# HELP genai_cache_hit_rate Cache hit rate for semantic cache",
        f"# TYPE genai_cache_hit_rate gauge",
        f"genai_cache_hit_rate {summary.get('cache_hit_rate', 0):.4f}",
        f"",
        f"# HELP genai_avg_confidence Average RAG confidence score",
        f"# TYPE genai_avg_confidence gauge",
        f"genai_avg_confidence {summary.get('avg_confidence', 0):.4f}",
        f"",
        f"# HELP genai_avg_hallucination_risk Average hallucination risk heuristic",
        f"# TYPE genai_avg_hallucination_risk gauge",
        f"genai_avg_hallucination_risk {summary.get('avg_hallucination_risk', 0):.4f}",
    ]

    for name, lat in summary.get("latencies_ms", {}).items():
        safe = name.replace("-", "_").replace(".", "_")
        lines += [
            f"",
            f"# HELP genai_latency_{safe}_p95_ms P95 latency in milliseconds for {name}",
            f"# TYPE genai_latency_{safe}_p95_ms gauge",
            f"genai_latency_{safe}_p95_ms {lat['p95']:.2f}",
        ]

    return "\n".join(lines)
