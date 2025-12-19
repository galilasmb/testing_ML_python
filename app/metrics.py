"""
Simple metrics tracking for monitoring service performance.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List
from threading import Lock


@dataclass
class Metrics:
    """Thread-safe metrics collector."""
    
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_inference_time_ms: float = 0.0
    latencies: List[float] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    _lock: Lock = field(default_factory=Lock, repr=False)
    
    def record_request(self, success: bool, inference_time_ms: float):
        """Record a request with its result and timing."""
        with self._lock:
            self.total_requests += 1
            if success:
                self.successful_requests += 1
            else:
                self.failed_requests += 1
            
            if inference_time_ms > 0:
                self.total_inference_time_ms += inference_time_ms
                self.latencies.append(inference_time_ms)
                
                # Keep only last 1000 latencies to avoid memory issues
                if len(self.latencies) > 1000:
                    self.latencies = self.latencies[-1000:]
    
    def get_stats(self) -> dict:
        """Get current metrics statistics."""
        with self._lock:
            uptime = (datetime.now() - self.start_time).total_seconds()
            
            avg_latency = 0.0
            p95_latency = 0.0
            p99_latency = 0.0
            
            if self.latencies:
                avg_latency = sum(self.latencies) / len(self.latencies)
                sorted_latencies = sorted(self.latencies)
                p95_idx = int(len(sorted_latencies) * 0.95)
                p99_idx = int(len(sorted_latencies) * 0.99)
                p95_latency = sorted_latencies[p95_idx] if p95_idx < len(sorted_latencies) else 0.0
                p99_latency = sorted_latencies[p99_idx] if p99_idx < len(sorted_latencies) else 0.0
            
            return {
                "uptime_seconds": round(uptime, 2),
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "success_rate": round(
                    self.successful_requests / self.total_requests * 100, 2
                ) if self.total_requests > 0 else 0.0,
                "average_latency_ms": round(avg_latency, 2),
                "p95_latency_ms": round(p95_latency, 2),
                "p99_latency_ms": round(p99_latency, 2),
                "requests_per_second": round(
                    self.total_requests / uptime, 2
                ) if uptime > 0 else 0.0,
            }
    
    def reset(self):
        """Reset all metrics."""
        with self._lock:
            self.total_requests = 0
            self.successful_requests = 0
            self.failed_requests = 0
            self.total_inference_time_ms = 0.0
            self.latencies.clear()
            self.start_time = datetime.now()


# Global metrics instance
metrics = Metrics()
