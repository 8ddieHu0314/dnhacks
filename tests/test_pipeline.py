import asyncio
import unittest

from vision_api.models import FrameMetadata
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
