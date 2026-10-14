"""Privacy-safe, process-local aggregate API instrumentation."""

from collections import defaultdict
from threading import Lock
from typing import Dict


KNOWN_ENDPOINTS = frozenset(
    {
        "/health", "/system-status", "/metrics", "/evaluation-report",
        "/symptoms", "/predict", "/explain", "/evidence",
        "/explain-grounded", "/next-question",
    }
)
ABSTENTION_REASONS = frozenset(
    {"low_top_probability", "low_top_two_separation", "high_distribution_entropy"}
)
LLM_OUTCOMES = frozenset({"generated", "unavailable", "failed", "skipped"})


class InMemoryObservability:
    """Store bounded aggregate counters only; never accept request payloads."""

    def __init__(self):
        self._lock = Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self._requests: Dict[str, Dict] = {}
            self._llm = {"generated": 0, "unavailable": 0, "failed": 0, "skipped": 0, "duration_count": 0, "duration_total_seconds": 0.0, "duration_max_seconds": 0.0}
            self._retrieval = {"requests": 0, "errors": 0, "passages_returned": 0, "duration_count": 0, "duration_total_seconds": 0.0, "duration_max_seconds": 0.0}
            self._predictions = {"count": 0, "abstained": 0, "reasons": {reason: 0 for reason in sorted(ABSTENTION_REASONS)}}

    def record_request(self, endpoint: str, status_code: int, duration_seconds: float) -> None:
        label = endpoint if endpoint in KNOWN_ENDPOINTS else "unmatched"
        status = str(int(status_code)) if 100 <= int(status_code) <= 599 else "other"
        elapsed = max(0.0, float(duration_seconds))
        with self._lock:
            item = self._requests.setdefault(label, {"requests": 0, "errors": 0, "status_counts": {}, "duration_count": 0, "duration_total_seconds": 0.0, "duration_max_seconds": 0.0})
            item["requests"] += 1
            item["errors"] += int(int(status_code) >= 400)
            item["status_counts"][status] = item["status_counts"].get(status, 0) + 1
            item["duration_count"] += 1
            item["duration_total_seconds"] += elapsed
            item["duration_max_seconds"] = max(item["duration_max_seconds"], elapsed)

    def record_retrieval(self, duration_seconds: float, passage_count: int = 0, failed: bool = False) -> None:
        elapsed = max(0.0, float(duration_seconds))
        with self._lock:
            item = self._retrieval
            item["requests"] += 1
            item["errors"] += int(bool(failed))
            item["passages_returned"] += max(0, int(passage_count))
            item["duration_count"] += 1
            item["duration_total_seconds"] += elapsed
            item["duration_max_seconds"] = max(item["duration_max_seconds"], elapsed)

    def record_prediction(self, prediction: dict) -> None:
        with self._lock:
            self._predictions["count"] += 1
            if prediction.get("insufficient_evidence") or prediction.get("decision_status") == "insufficient_evidence":
                self._predictions["abstained"] += 1
                for reason in prediction.get("abstention_reasons", []):
                    label = reason if reason in ABSTENTION_REASONS else "other"
                    if label == "other":
                        self._predictions["reasons"].setdefault("other", 0)
                    self._predictions["reasons"][label] += 1

    def record_llm(self, outcome: str, duration_seconds: float = 0.0) -> None:
        label = outcome if outcome in LLM_OUTCOMES else "failed"
        elapsed = max(0.0, float(duration_seconds))
        with self._lock:
            self._llm[label] += 1
            if label != "skipped":
                self._llm["duration_count"] += 1
                self._llm["duration_total_seconds"] += elapsed
                self._llm["duration_max_seconds"] = max(self._llm["duration_max_seconds"], elapsed)

    @staticmethod
    def _durations(item: dict) -> dict:
        count = item["duration_count"]
        return {
            "count": count,
            "mean_ms": round(item["duration_total_seconds"] * 1000 / count, 3) if count else None,
            "max_ms": round(item["duration_max_seconds"] * 1000, 3) if count else None,
        }

    def snapshot(self) -> dict:
        with self._lock:
            by_endpoint = {}
            for endpoint, item in self._requests.items():
                by_endpoint[endpoint] = {
                    "requests": item["requests"],
                    "errors": item["errors"],
                    "status_counts": dict(item["status_counts"]),
                    "latency": self._durations(item),
                }
            requests_total = sum(item["requests"] for item in self._requests.values())
            errors_total = sum(item["errors"] for item in self._requests.values())
            return {
                "storage": "in_memory_since_process_start",
                "privacy": "Aggregate counters only; request bodies, query strings, and symptom values are not retained.",
                "requests_total": requests_total,
                "errors_total": errors_total,
                "by_endpoint": by_endpoint,
                "request_latency": self._durations({
                    "duration_count": sum(item["duration_count"] for item in self._requests.values()),
                    "duration_total_seconds": sum(item["duration_total_seconds"] for item in self._requests.values()),
                    "duration_max_seconds": max((item["duration_max_seconds"] for item in self._requests.values()), default=0.0),
                }),
                "retrieval": {"requests": self._retrieval["requests"], "errors": self._retrieval["errors"], "passages_returned": self._retrieval["passages_returned"], "latency": self._durations(self._retrieval)},
                "llm_generation": {"outcomes": {key: self._llm[key] for key in ("generated", "unavailable", "failed", "skipped")}, "duration": self._durations(self._llm)},
                "predictions": {"count": self._predictions["count"], "abstained": self._predictions["abstained"], "abstention_reasons": dict(self._predictions["reasons"])},
            }


OBSERVABILITY = InMemoryObservability()
