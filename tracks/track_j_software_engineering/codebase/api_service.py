"""
API Gateway & Token Rate Limiter.
Module of realistic production codebase for Track J software optimization.
Student: Ayaan Mirza (Roll No: 241070043)
"""

import time
from typing import Callable, Any

class TokenBucketRateLimiter:
    def __init__(self, capacity: int = 10, refill_rate_per_sec: float = 2.0):
        self.capacity = capacity
        self.refill_rate = refill_rate_per_sec
        self.tokens = float(capacity)
        self.last_update = time.time()

    def consume(self, cost: int = 1) -> bool:
        now = time.time()
        elapsed = now - self.last_update
        self.tokens = min(float(self.capacity), self.tokens + elapsed * self.refill_rate)
        self.last_update = now

        if self.tokens >= cost:
            self.tokens -= cost
            return True
        return False

class APIRequest:
    def __init__(self, endpoint: str, client_id: str, payload: dict[str, Any]):
        self.endpoint = endpoint
        self.client_id = client_id
        self.payload = payload
        self.timestamp = time.time()

class APIRouter:
    def __init__(self):
        self.routes: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {}
        self.limiters: dict[str, TokenBucketRateLimiter] = {}
        self.total_handled = 0
        self.total_rejected = 0

    def register_endpoint(self, endpoint: str, handler: Callable[[dict[str, Any]], dict[str, Any]]):
        self.routes[endpoint] = handler

    def dispatch(self, req: APIRequest) -> dict[str, Any]:
        if req.client_id not in self.limiters:
            self.limiters[req.client_id] = TokenBucketRateLimiter(capacity=5, refill_rate_per_sec=10.0)
        limiter = self.limiters[req.client_id]

        if not limiter.consume(1):
            self.total_rejected += 1
            return {'status': 429, 'error': 'Rate limit exceeded'}

        if req.endpoint not in self.routes:
            return {'status': 404, 'error': 'Endpoint not found'}

        try:
            handler = self.routes[req.endpoint]
            data = handler(req.payload)
            self.total_handled += 1
            return {'status': 200, 'data': data}
        except Exception as exc:
            return {'status': 500, 'error': str(exc)}
