from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from datetime import datetime, timezone

from .models import Frame, FrameMetadata, SegmentationResult, SessionMetrics
from .segmentation import SegmentationEngine


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class VisionPipeline:
    """Bounded, latest-frame-first processing pipeline.

    Live safety guidance should favor the newest point-of-view frame over perfect
    delivery of old frames. When full, the pipeline drops the oldest queued frame.
    This is intentional backpressure, not packet loss hidden from callers.
    """

    def __init__(
        self,
        engine: SegmentationEngine,
        *,
        queue_capacity: int,
        result_history: int,
    ) -> None:
        self._engine = engine
        self._queue: asyncio.Queue[Frame] = asyncio.Queue(maxsize=queue_capacity)
        self._results: dict[str, deque[SegmentationResult]] = defaultdict(
            lambda: deque(maxlen=result_history)
        )
        self._received: dict[str, int] = defaultdict(int)
        self._processed: dict[str, int] = defaultdict(int)
        self._dropped: dict[str, int] = defaultdict(int)
        self._worker: asyncio.Task[None] | None = None
        self._stopping = asyncio.Event()

    @property
    def backend_name(self) -> str:
        return self._engine.name

    async def start(self) -> None:
        if self._worker is None:
            self._stopping.clear()
            self._worker = asyncio.create_task(self._run(), name="segmentation-worker")

    async def stop(self) -> None:
        self._stopping.set()
        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
            self._worker = None
