import time
from collections import OrderedDict

SENTINEL = "skip_thought_signature_validator"


class SignatureCache:
    def __init__(self, maximum: int = 4096, ttl: float = 7200):
        self.maximum, self.ttl = maximum, ttl
        self.entries: OrderedDict[tuple[str, str], tuple[str, str, float]] = OrderedDict()

    def put(self, kind: str, key: str, signature: str, family: str) -> None:
        if (
            not key
            or not isinstance(signature, str)
            or len(signature) < 50
            or signature == SENTINEL
        ):
            return
        self.entries[kind, key] = signature, family, time.monotonic() + self.ttl
        self.entries.move_to_end((kind, key))
        while len(self.entries) > self.maximum:
            self.entries.popitem(last=False)

    def get(self, kind: str, key: str, family: str) -> str | None:
        item = self.entries.get((kind, key))
        if item and item[2] > time.monotonic() and item[1] == family:
            return item[0]
        if item and item[2] <= time.monotonic():
            self.entries.pop((kind, key), None)
        return None

    def clear(self) -> None:
        self.entries.clear()

    def __len__(self) -> int:
        now = time.monotonic()
        for key in [k for k, value in self.entries.items() if value[2] <= now]:
            del self.entries[key]
        return len(self.entries)
