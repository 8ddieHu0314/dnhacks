import asyncio
import unittest

from vision_api.models import FrameMetadata, VisionOutput
from vision_api.pipeline import VisionPipeline
from vision_api.segmentation import MockSegmentationEngine


class VisionPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.pipeline = VisionPipeline(
            MockSegmentationEngine(), queue_capacity=2, result_history=3
        )
        await self.pipeline.start()

    async def asyncTearDown(self) -> None:
        await self.pipeline.stop()

    async def test_processes_a_frame_with_the_contract_shape(self) -> None:
        metadata = FrameMetadata(frame_id="frame-1", width=640, height=480)
        await self.pipeline.submit("session-1", metadata, b"not-a-real-jpeg-yet")
        await asyncio.wait_for(self.pipeline._queue.join(), timeout=1)

        results = self.pipeline.results_for("session-1")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].frame_id, "frame-1")
        self.assertEqual(results[0].backend, "mock")
        self.assertEqual(results[0].regions[0].label, "mock-region")

    async def test_notifies_a_result_handler(self) -> None:
        received = []

        async def collect(result) -> None:
            received.append(result.frame_id)

        pipeline = VisionPipeline(MockSegmentationEngine(), queue_capacity=1, result_history=1, on_result=collect)
        await pipeline.start()
        try:
            await pipeline.submit("session", FrameMetadata(frame_id="spoken", width=2, height=2), b"frame")
            await asyncio.wait_for(pipeline._queue.join(), timeout=1)
            self.assertEqual(received, ["spoken"])
        finally:
            await pipeline.stop()

    async def test_records_a_model_failure_without_stopping_the_worker(self) -> None:
        class FailingEngine:
            name = "failing"

            async def analyze(self, _frame):
                raise RuntimeError("model unavailable")

        pipeline = VisionPipeline(FailingEngine(), queue_capacity=1, result_history=1)
        await pipeline.start()
        try:
            metadata = FrameMetadata(frame_id="failed", width=2, height=2)
            with self.assertLogs("vision_api.pipeline", level="ERROR"):
                await pipeline.submit("session", metadata, b"frame")
                await asyncio.wait_for(pipeline._queue.join(), timeout=1)
            self.assertEqual(pipeline.metrics_for("session").failed_frames, 1)
        finally:
            await pipeline.stop()

    async def test_drops_queued_frames_in_favor_of_the_newest_view(self) -> None:
        class BlockingEngine:
            name = "blocking"

            def __init__(self) -> None:
                self.started, self.release = asyncio.Event(), asyncio.Event()

            async def analyze(self, _frame):
                self.started.set()
                await self.release.wait()
                return VisionOutput()

        engine = BlockingEngine()
        pipeline = VisionPipeline(engine, queue_capacity=3, result_history=3)
        await pipeline.start()
        try:
            await pipeline.submit("session", FrameMetadata(frame_id="first", width=2, height=2), b"1")
            await asyncio.wait_for(engine.started.wait(), timeout=1)
            await pipeline.submit("session", FrameMetadata(frame_id="stale", width=2, height=2), b"2")
            await pipeline.submit("session", FrameMetadata(frame_id="newest", width=2, height=2), b"3")
            engine.release.set()
            await asyncio.wait_for(pipeline._queue.join(), timeout=1)
            self.assertEqual([result.frame_id for result in pipeline.results_for("session")], ["first", "newest"])
            self.assertEqual(pipeline.metrics_for("session").dropped_stale_frames, 1)
        finally:
            await pipeline.stop()
