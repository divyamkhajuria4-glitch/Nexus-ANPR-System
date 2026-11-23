import shutil
import tempfile
import unittest
from pathlib import Path

import cv2

from tests.stubs import FPS, H, W, SynthVehicle, make_video
from utils.video_utils import (VideoError, create_writer, iter_frames, open_capture, probe_video,
                               ensure_output_space, estimated_video_space_bytes, transcode_h264)


class VideoUtilsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(self.tmp, ignore_errors=True))
        self.video = make_video(self.tmp / "v.mp4", [SynthVehicle(1, "MH12AB1234", 0, 19, 10)], 20)

    def test_probe_video(self):
        info = probe_video(self.video)
        self.assertEqual((info.width, info.height), (W, H))
        self.assertAlmostEqual(info.fps, FPS, places=1)
        self.assertEqual(info.frame_count, 20)

    def test_probe_rejects_missing_and_corrupt(self):
        with self.assertRaises(VideoError):
            probe_video(self.tmp / "missing.mp4")
        bad = self.tmp / "bad.mp4"; bad.write_bytes(b"not a video at all")
        with self.assertRaises(VideoError):
            probe_video(bad)

    def test_frame_extraction_is_sequential_and_complete(self):
        frames = list(iter_frames(self.video))
        self.assertEqual([i for i, _ in frames], list(range(20)))
        self.assertEqual(frames[0][1].shape, (H, W, 3))
        self.assertEqual(len(list(iter_frames(self.video, max_frames=5))), 5)

    def test_open_capture_releases(self):
        with open_capture(self.video) as cap:
            self.assertTrue(cap.isOpened())
        self.assertFalse(cap.isOpened())

    def test_writer_roundtrip(self):
        out = self.tmp / "o.mp4"
        w = create_writer(out, 10.0, (64, 48))
        for _ in range(7):
            w.write(cv2.UMat(48, 64, cv2.CV_8UC3).get() * 0 + 50)
        w.release()
        self.assertEqual(len(list(iter_frames(out))), 7)

    def test_writer_failure_is_reported(self):
        with self.assertRaises(VideoError):
            create_writer(self.tmp / "x.mp4", 10.0, (0, 0))

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg not installed")
    def test_h264_transcode_is_playable(self):
        dst = self.tmp / "h264.mp4"
        self.assertTrue(transcode_h264(self.video, dst))
        self.assertEqual(len(list(iter_frames(dst))), 20)

    def test_transcode_failure_returns_false(self):
        self.assertFalse(transcode_h264(self.tmp / "nope.mp4", self.tmp / "out.mp4"))
        self.assertFalse((self.tmp / ".out.partial.mp4").exists())

    def test_output_space_preflight_rejects_unavailable_capacity(self):
        need = estimated_video_space_bytes(3840, 2160, 1632)
        self.assertGreater(need, estimated_video_space_bytes(64, 48, 20))
        with self.assertRaises(VideoError):
            ensure_output_space(self.tmp, 10**20)


if __name__ == "__main__":
    unittest.main()
