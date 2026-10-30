"""
Cache Replacement Policies and In-Memory Storage Engines.
Module of realistic production codebase for Track J software optimization.
Student: Ayaan Mirza (Roll No: 241070043)
"""

import time
from typing import Any, Optional
from collections import OrderedDict

class LRUCache:
    """Least Recently Used (LRU) Cache policy with O(1) lookups and evictions."""
    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.cache: OrderedDict[str, Any] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        if key not in self.cache:
            self.misses += 1
            return None
        self.hits += 1
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: str, value: Any):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return (self.hits / total) if total > 0 else 0.0

class LFUCache:
    """Least Frequently Used (LFU) Cache policy."""
    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.values: dict[str, Any] = {}
        self.counts: dict[str, int] = {}
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        if key not in self.values:
            self.misses += 1
            return None
        self.hits += 1
        self.counts[key] += 1
        return self.values[key]

    def put(self, key: str, value: Any):
        if self.capacity <= 0:
            return
        if key in self.values:
            self.values[key] = value
            self.counts[key] += 1
            return

        if len(self.values) >= self.capacity:
            # Find key with minimum count
            min_key = min(self.counts, key=self.counts.get)
            del self.values[min_key]
            del self.counts[min_key]

        self.values[key] = value
        self.counts[key] = 1

    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return (self.hits / total) if total > 0 else 0.0

class TTLCache:
    """Time-To-Live (TTL) Cache policy with automatic timestamp invalidation."""
    def __init__(self, default_ttl_sec: float = 60.0):
        self.default_ttl = default_ttl_sec
        self.store: dict[str, tuple[Any, float]] = {}

    def set(self, key: str, val: Any, ttl: Optional[float] = None):
        expiry = time.time() + (ttl if ttl is not None else self.default_ttl)
        self.store[key] = (val, expiry)

    def get(self, key: str) -> Optional[Any]:
        if key not in self.store:
            return None
        val, expiry = self.store[key]
        if time.time() > expiry:
            del self.store[key]
            return None
        return val

    def purge_expired(self) -> int:
        now = time.time()
        expired_keys = [k for k, (_, exp) in self.store.items() if now > exp]
        for k in expired_keys:
            del self.store[k]
        return len(expired_keys)
