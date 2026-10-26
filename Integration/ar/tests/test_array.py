"""Step 50 checks: array geometry, near-plane clipping, depth-ordered compositing, occlusion over the array, gesture mapping."""
from pathlib import Path
import unittest
import numpy as np
from ar_projection.array import ArrayState, PanelSet, clip_near, local_corners, world_corners
from ar_projection.config import Array, Gestures, load_config
from ar_projection.geometry import inverse, transform
from ar_projection.gestures import EdgeTrigger, action_for
from ar_projection.occlusion import panel_layer
from ar_projection.rendering import visible_panel
from ar_projection.types import CameraModel

ROOT = Path(__file__).resolve().parents[1]
CAM = CameraModel(np.array([[600., 0, 320], [0, 600., 240], [0, 0, 1]]), np.zeros(5), (640, 480), {"size": [640, 480], "undistort": True})
OFFSET = 0.15


def look_at(eye, target, up=(0, 0, 1.)):
    """T_CW for a camera at `eye` looking at `target` (camera x right, y down, z forward)."""
    z = np.asarray(target, float)-eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    R = np.stack((x, np.cross(z, x), z))
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, -R @ eye
    return T


def centre(P):
    return P.mean(axis=0)


class ArrayGeometry(unittest.TestCase):
    cfg = Array(enabled=True)
    eye = np.array([0., 2., 1.2])               # the live runs put the camera on the +Y side of the tag

    def test_layout_at_scale_one(self):
        P = world_corners(self.cfg, ArrayState(), OFFSET)
        g = P["game"]
        self.assertAlmostEqual(np.linalg.norm(g[1]-g[0]), 0.5)
        self.assertAlmostEqual(np.linalg.norm(g[3]-g[0]), 0.28125)
        self.assertTrue(np.allclose(g[:2, 2], 0) and np.allclose(g[:2, 1], OFFSET))      # base on the floor at the pivot offset
        self.assertAlmostEqual(np.linalg.norm(P["video"][1]-g[0]), 0.05, places=6)       # inner edge 5 cm from the centre panel
        self.assertAlmostEqual(np.linalg.norm(P["stats"][0]-g[1]), 0.05, places=6)
        self.assertAlmostEqual(P["score"][0][2], 0.1875+0.05)                            # top panel 5 cm above the bottom one
        self.assertAlmostEqual(np.linalg.norm(P["stats"][1]-P["stats"][0]), 0.25)
        self.assertAlmostEqual(np.linalg.norm(P["video"][3]-P["video"][0]), 0.4444)

    def test_corner_order_is_bottom_left_first_for_the_viewer(self):
        T = look_at(self.eye, np.array([0., OFFSET, 0.2]))
        for n, P in world_corners(self.cfg, ArrayState(yaw_deg=30), OFFSET).items():
            uv = (P @ T[:3, :3].T+T[:3, 3]) @ CAM.K.T
            uv = uv[:, :2]/uv[:, 2:3]
            self.assertLess(uv[0, 0], uv[1, 0], n)       # corner 0 is left of corner 1
            self.assertGreater(uv[0, 1], uv[3, 1], n)    # and below corner 3 (image y grows downward)

    def test_toe_in_turns_left_panel_to_viewer(self):
        P = world_corners(self.cfg, ArrayState(), OFFSET)["video"]
        inner, outer = P[1], P[0]
        self.assertAlmostEqual(np.degrees(np.arctan2(abs(outer[1]-inner[1]), abs(outer[0]-inner[0]))), 5.0, places=4)
        self.assertGreater(outer[1], inner[1])                                           # outer end nearer the +Y viewer

    def test_scale_and_yaw_about_floor_pivot(self):
        s = ArrayState(yaw_deg=30, scale=2.0)
        one, two = world_corners(self.cfg, ArrayState(yaw_deg=30), OFFSET), world_corners(self.cfg, s, OFFSET)
        pivot = np.array([0, OFFSET, 0])
        for n in one:
            self.assertTrue(np.allclose(two[n]-pivot, 2*(one[n]-pivot)))
        self.assertTrue(all(np.allclose(P[:, 2] >= -1e-12, True) for P in two.values()))
        self.assertTrue(np.allclose(np.mean([P[:2] for P in world_corners(self.cfg, ArrayState(yaw_deg=60), OFFSET).values()], axis=(0, 1))[2] >= 0, True))
        # the centre panel's bottom-edge midpoint is the pivot at any yaw/scale
        self.assertTrue(np.allclose(two["game"][:2].mean(axis=0), pivot))

    def test_limits_reset_and_game_wrap(self):
        s, c = ArrayState(), self.cfg
        for _ in range(10):
            s.apply(c, "yaw", 1)
        self.assertEqual(s.yaw_deg, 60.0)
        for _ in range(30):
            s.apply(c, "yaw", -1)
        self.assertEqual(s.yaw_deg, -60.0)
        for _ in range(30):
            s.apply(c, "scale", 1)
        self.assertAlmostEqual(s.scale, 2.0)
        for _ in range(60):
            s.apply(c, "scale", -1)
        self.assertAlmostEqual(s.scale, 0.5)
        s.apply(c, "reset")
        self.assertEqual((s.yaw_deg, s.scale), (0.0, 1.0))
        s.apply(c, "game", -1)
        self.assertEqual(s.game, len(c.games)-1)
        s.apply(c, "stop")
        self.assertFalse(s.playing)
        s.apply(c, "play")
        self.assertTrue(s.playing)

    def test_yaw_sign_clockwise_brings_left_panel_closer(self):
        gestures = Gestures()
        action, value = action_for(gestures.actions, "Turning Hand Clockwise")
        s = ArrayState()
        s.apply(self.cfg, action, value)
        P = world_corners(self.cfg, s, OFFSET)
        self.assertLess(np.linalg.norm(centre(P["video"])-self.eye), np.linalg.norm(centre(P["stats"])-self.eye))
        action, value = action_for(gestures.actions, "Turning Hand Counterclockwise")
        s = ArrayState()
        s.apply(self.cfg, action, value)
        P = world_corners(self.cfg, s, OFFSET)
        self.assertGreater(np.linalg.norm(centre(P["video"])-self.eye), np.linalg.norm(centre(P["stats"])-self.eye))


class Clipping(unittest.TestCase):
    quad = np.array([[-1, 0, 0], [1, 0, 0], [1, 0, 1], [-1, 0, 1.]])

    def test_partly_behind_is_clipped_not_dropped(self):
        T = look_at(np.array([0., -0.2, 0.5]), np.array([0., 0.3, 0.]))      # camera pitched down at a floor quad that passes under it
        quad = np.array([[-1, 0.5, 0], [1, 0.5, 0], [1, -0.5, 0], [-1, -0.5, 0.]])
        clipped = clip_near(T, quad, 0.05)
        self.assertIsNotNone(clipped)
        self.assertTrue(np.all(clipped[:, 2] >= 0.05-1e-9) and len(clipped) >= 3)
        self.assertIsNotNone(visible_panel(T, quad, CAM))

    def test_fully_behind_is_none(self):
        T = look_at(np.array([0., 5., 1.]), np.array([0., 10., 1.]))
        self.assertIsNone(clip_near(T, self.quad, 0.05))
        self.assertIsNone(visible_panel(T, self.quad, CAM))

    def test_front_panel_unchanged(self):
        T = look_at(np.array([0., 3., 0.5]), np.array([0., 0., 0.5]))
        self.assertEqual(len(clip_near(T, self.quad, 0.05)), 4)


class Compositing(unittest.TestCase):
    T = look_at(np.array([0., 3., 0.5]), np.array([0., 0., 0.5]))

    def quad(self, y, half=0.4):
        return np.array([[-half, y, 0.1], [half, y, 0.1], [half, y, 0.9], [-half, y, 0.9]])

    def textures(self):
        return {"near": np.full((10, 10, 3), (0, 0, 255), np.uint8), "far": np.full((10, 10, 3), (255, 0, 0), np.uint8)}

    def test_nearest_panel_wins_and_orientation(self):
        ps = PanelSet({"far": self.quad(0.0, 0.6), "near": self.quad(1.0, 0.2)}, self.textures(), "far")
        color, visible, z = panel_layer(self.T, ps, CAM)
        mid = (240, 320)
        self.assertTrue(visible[mid])
        self.assertAlmostEqual(z[mid], 2.0, places=3)                                  # near panel at y=1 seen from y=3
        self.assertTrue(np.array_equal(color[mid], [0, 0, 255]))
        self.assertTrue(np.array_equal(color[240, 230], [255, 0, 0]))                  # far panel outside the near one
        self.assertEqual(ps.names[ps.ids[mid]], "near")

    def test_texture_is_upright_and_not_mirrored_for_viewer(self):
        # viewer on +Y: panel corner 0 is the viewer's bottom-left only if the quad is listed in viewer order (right = -X)
        P = np.array([[0.4, 0, 0.1], [-0.4, 0, 0.1], [-0.4, 0, 0.9], [0.4, 0, 0.9]])
        tex = np.zeros((20, 20, 3), np.uint8)
        tex[:10, :10] = (0, 255, 0)                                                      # top-left of the texture
        color, visible, _ = panel_layer(self.T, PanelSet({"p": P}, {"p": tex}, "p"), CAM)
        ys, xs = np.nonzero(visible & (color[..., 1] == 255))
        cy, cx = np.nonzero(visible)[0].mean(), np.nonzero(visible)[1].mean()
        self.assertLess(ys.mean(), cy)
        self.assertLess(xs.mean(), cx)

    def test_panel_partly_behind_camera_renders(self):
        eye = np.array([0., 0.3, 0.5])
        T = look_at(eye, np.array([0., -1., 0.5]))
        P = np.array([[-1, 1.0, 0], [1, 1.0, 0], [1, -1.0, 0], [-1, -1.0, 0.]])       # floor-like quad passing under/behind the camera
        ps = PanelSet({"p": P}, {"p": np.full((8, 8, 3), 200, np.uint8)}, "p")
        color, visible, z = panel_layer(T, ps, CAM)
        self.assertTrue(visible.any() and np.all(z[visible] >= 0.05))

    def test_occlusion_hides_pixels_nearer_than_margin(self):
        ps = PanelSet({"near": self.quad(1.0, 0.3)}, self.textures(), "near")
        scene = np.full((480, 640), 5.0)
        scene[:, :320] = 1.0                                                              # an occluder 1 m from the camera, panel is at 2 m
        _, visible, z = panel_layer(self.T, ps, CAM, scene, margin_m=0.1, margin_fraction=0.1)
        self.assertFalse(visible[240, 300])
        self.assertTrue(visible[240, 340])
        scene[:, :320] = 1.95                                                             # within the 0.2 m margin: still visible
        _, visible, _ = panel_layer(self.T, ps, CAM, scene, margin_m=0.1, margin_fraction=0.1)
        self.assertTrue(visible[240, 300])


class GestureMapping(unittest.TestCase):
    def test_table(self):
        t = Gestures().actions
        self.assertEqual(action_for(t, "Swiping Left"), ("game", -1))
        self.assertEqual(action_for(t, "Swiping Right"), ("game", 1))
        self.assertEqual(action_for(t, "Zooming In With Full Hand"), ("scale", 1))
        self.assertEqual(action_for(t, "Thumb Down"), ("reset", 0))
        self.assertEqual(action_for(t, "Stop Sign"), ("stop", 0))
        self.assertEqual(action_for(t, "Thumb Up"), ("play", 1))
        self.assertIsNone(action_for(t, "No gesture"))

    def test_edge_trigger_and_cooldown(self):
        e = EdgeTrigger(1.0)
        self.assertIsNone(e.update(None, 0.0))
        self.assertEqual(e.update("Thumb Up", 1.0), "Thumb Up")
        self.assertIsNone(e.update("Thumb Up", 1.2))               # still the same confirmation
        self.assertIsNone(e.update(None, 1.5))
        self.assertIsNone(e.update("Thumb Up", 1.8))               # new confirmation inside the cooldown is dropped
        self.assertIsNone(e.update(None, 2.0))
        self.assertEqual(e.update("Stop Sign", 2.5), "Stop Sign")  # after the cooldown
        self.assertEqual(e.update("Thumb Up", 4.0), "Thumb Up")    # a changed label fires too


class TuningRequests(unittest.TestCase):
    def test_request_only_when_changed(self):
        from ar_projection.gestures import GestureWorker
        w = GestureWorker(Gestures(), None)
        w.request(0.70, 5, 4)
        self.assertIsNone(w.pending)
        w.request(0.55, 6, 4)
        self.assertEqual(w.pending, (0.55, 6, 4))


class ConfigLoads(unittest.TestCase):
    def test_old_config_has_array_and_gestures_off(self):
        c = load_config(Path(__file__).parent/"fixtures"/"tracking-config.json")
        self.assertFalse(c.array.enabled or c.gestures.enabled)

    def test_array_config(self):
        c = load_config(ROOT/"config-array.json")
        self.assertTrue(c.array.enabled and c.gestures.enabled)
        self.assertEqual((c.array.yaw_step_deg, c.array.yaw_limit_deg, c.array.scale_max, c.array.scale_min), (15.0, 60.0, 2.0, 0.5))


if __name__ == "__main__":
    unittest.main()
