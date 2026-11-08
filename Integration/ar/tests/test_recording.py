"""Step 52 checks: panel videos follow the frame's source time (so a --replay-run shows the live run's panel frames), the
per-frame panel frame log, the games-manifest loading fix, and the recording compositor's time alignment."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from ar_projection.config import Array, load_config
from ar_projection.content import ArrayContent, VideoLoop
from ar_projection.array import ArrayState

ROOT = Path(__file__).resolve().parents[1]


def numbered_video(path, n=50, fps=25):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (64, 48))
    for i in range(n):
        writer.write(np.full((48, 64, 3), i*4, np.uint8))
    writer.release()


class SourceClock(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.video = Path(self.tmp.name)/"v.avi"
        numbered_video(self.video)

    def tearDown(self):
        self.tmp.cleanup()

    def test_video_loop_follows_the_given_clock_not_the_wall_clock(self):
        loop = VideoLoop(self.video)
        self.assertIsNone(loop.index)
        loop.update(True, 5000.0)                 # an arbitrary source-time epoch, far from time.monotonic()
        self.assertEqual(loop.index, 0)
        loop.update(True, 5000.4)                 # 0.4 s at 25 fps
        self.assertEqual(loop.index, 10)
        loop.update(False, 5001.0)                # paused: frozen
        self.assertEqual(loop.index, 10)
        loop.update(True, 5002.0)                 # resumed: the paused second is skipped (the clock stopped at 5001.0, frame 25)
        self.assertEqual(loop.index, 25)
        loop.update(True, 5002.2)
        self.assertEqual(loop.index, 30)
        loop.update(True, 5004.0)                 # past the end (50 frames = 2 s): wraps to the start
        self.assertLess(loop.index, 50)
        loop.close()

    def test_two_runs_on_the_same_timestamps_show_the_same_frames(self):
        cfg = Array(enabled=True, games=[{"name": "A", "video": str(self.video), "topdown": str(self.video)},
                                         {"name": "B", "video": str(self.video), "topdown": str(self.video)}])
        logs = []
        for _ in range(2):
            content, state, log = ArrayContent(cfg), ArrayState(), []
            for k, t in enumerate(np.arange(7000.0, 7001.6, 0.093)):
                if k == 9:
                    state.apply(cfg, "game", 1)
                content.textures(state, float(t))
                log.append(json.dumps(content.frames))
            content.close()
            logs.append(log)
        self.assertEqual(logs[0], logs[1])
        frames = [json.loads(x) for x in logs[0]]
        self.assertEqual(frames[0]["game"], [0, 0])
        self.assertEqual(frames[9]["game"], [1, 0])          # a swipe restarts the new game's video at frame 0
        self.assertEqual(frames[9]["left"][:2], [1, 0])


class ManifestLoading(unittest.TestCase):
    def test_config_array_loads_the_games_manifest(self):
        c = load_config(ROOT/"config-array.json")
        raw = json.loads((ROOT/"config-array.json").read_text(encoding="utf-8"))["array"]
        manifest = (ROOT/raw["games_manifest"]).resolve()
        if "games" in raw or not manifest.exists():
            self.skipTest("config-array.json has explicit games or no manifest")
        self.assertEqual([g["name"] for g in c.array.games], [g["name"] for g in json.loads(manifest.read_text(encoding="utf-8"))])


class Compositor(unittest.TestCase):
    def test_sample_and_hold_on_logged_times(self):
        sys.path.insert(0, str(ROOT/"scripts"))
        import rec_video
        with tempfile.TemporaryDirectory() as tmp:
            rows = [{"frame_index": i, "source_time_s": 100+t} for i, t in enumerate((0, .1, .35, .4, 1.0))]
            (Path(tmp)/"frames.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows), encoding="utf-8")
            run = rec_video.Run(tmp)
            self.assertEqual([run.index_at(t) for t in (0, .05, .1, .34, .36, .99, 5)], [0, 0, 1, 1, 2, 3, 4])
        out = rec_video.fit(np.zeros((100, 200, 3), np.uint8)+255, (64, 64))
        self.assertEqual(out.shape, (64, 64, 3))
        self.assertEqual(int(out[0, 0, 0]), 0)               # letterboxed
        self.assertEqual(int(out[32, 32, 0]), 255)


if __name__ == "__main__":
    unittest.main()
