from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict, deque
from datetime import datetime, timezone

from .models import Frame, FrameMetadata, SegmentationResult, SessionMetrics, WorkflowDefinition
from .segmentation import VisionEngine

logger = logging.getLogger(__name__)


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
        engine: VisionEngine,
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
        self._failed: dict[str, int] = defaultdict(int)
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

    async def submit(
        self,
        session_id: str,
        metadata: FrameMetadata,
        image_bytes: bytes,
        workflow: WorkflowDefinition | None = None,
    ) -> int:
        self._received[session_id] += 1
        frame = Frame(
            session_id=session_id,
            metadata=metadata,
            image_bytes=image_bytes,
            received_at=utc_now(),
            workflow=workflow,
        )
        if self._queue.full():
            discarded = self._queue.get_nowait()
            self._queue.task_done()
            self._dropped[discarded.session_id] += 1
        self._queue.put_nowait(frame)
        return self._dropped[session_id]

    def results_for(self, session_id: str) -> list[SegmentationResult]:
        return list(self._results[session_id])

    def metrics_for(self, session_id: str) -> SessionMetrics:
        return SessionMetrics(
            session_id=session_id,
            received_frames=self._received[session_id],
            processed_frames=self._processed[session_id],
            failed_frames=self._failed[session_id],
            dropped_stale_frames=self._dropped[session_id],
            queue_depth=self._queue.qsize(),
        )

    async def _run(self) -> None:
        while not self._stopping.is_set():
            frame = await self._queue.get()
            started = time.perf_counter()
            try:
                output = await self._engine.analyze(frame)
                latency_ms = (time.perf_counter() - started) * 1_000
                self._results[frame.session_id].append(
                    SegmentationResult(
                        session_id=frame.session_id,
                        frame_id=frame.metadata.frame_id,
                        backend=self._engine.name,
                        workflow_id=frame.workflow.id if frame.workflow else None,
                        latency_ms=latency_ms,
                        regions=output.regions,
                        analysis=output.analysis,
                    )
                )
                self._processed[frame.session_id] += 1
            except asyncio.CancelledError:
                raise
            except Exception:
                self._failed[frame.session_id] += 1
                logger.exception("Vision engine failed for frame %s", frame.metadata.frame_id)
            finally:
                self._queue.task_done()
