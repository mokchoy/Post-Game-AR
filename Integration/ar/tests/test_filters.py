"""Step 51 pose filters: OneEuroPose, KalmanPose and the PoseSmoother interface."""
import json
import unittest
from pathlib import Path
import numpy as np
from ar_projection.config import load_config
from ar_projection.filters import KalmanPose, OneEuroPose, _join, _split, alpha, covariance_wc, exp, log, make_smoother
from ar_projection.rendering import PoseSmoother

CFG = Path(__file__).resolve().parent.parent/"config-array.json"
RUN = Path(__file__).resolve().parent.parent.parent/"runs/20261027T033608-9db092df"


class P:                                           # filter parameters, same names as config.Filter
    def __init__(self, **kw):
        self.__dict__.update(dict(enabled=True, measurement_weight=0.4, min_cutoff_hz=1.0, beta=0.0, d_cutoff_hz=1.0, scene_distance_m=2.0,
                                  q=1.0, r_scale=1.0, gate_chi2=22.46, measurement_sigma=[0.003]*3+[0.001]*3), **kw)


def pose(c, rot=(0, 0, 0)):
    return _join(np.array(c, float), exp(rot))


def move(T, T0):
    (a, Ra), (b, Rb) = _split(T), _split(T0)
    return np.linalg.norm(a-b), np.linalg.norm(log(Ra @ Rb.T))


class Filters(unittest.TestCase):
    def test_constant_input_converges(self):
        T = pose([0.3, -0.2, 1.5], (0.1, 0.2, -0.1))
        for f in (PoseSmoother(P()), OneEuroPose(P(beta=1.0)), KalmanPose(P(q=0.1))):
            out = pose([0, 0, 0])
            for k in range(200):
                out = f.update(T, k*0.1)
            self.assertLess(sum(move(out, T)), 1e-6)

    def test_ema_step_response_never_overshoots(self):
        f, T = PoseSmoother(P()), pose([1.0, 0, 0])
        f.update(pose([0, 0, 0]), 0.0)
        xs = [_split(f.update(T, 0.1*(k+1)))[0][0] for k in range(30)]
        self.assertTrue(np.all(np.diff(xs) >= 0) and max(xs) <= 1.0+1e-12)

    def test_kalman_constant_velocity_ramp_has_small_steady_state_error(self):
        f, v = KalmanPose(P(q=1.0)), np.array([0.2, 0.0, -0.1])
        for k in range(300):
            t = 0.1*k
            out = f.update(pose(v*t+[0, 0, 1.5]), t)
        self.assertLess(np.linalg.norm(_split(out)[0]-(v*t+[0, 0, 1.5])), 1e-3)

    def test_one_euro_without_beta_is_a_dt_aware_ema(self):
        a, b, rng = OneEuroPose(P(min_cutoff_hz=0.7)), None, np.random.default_rng(1)
        t, c, R = 0.0, None, None
        for k in range(40):
            t += rng.uniform(0.05, 0.15)
            T = pose(rng.normal(size=3)*0.01+[0, 0, 1.5], rng.normal(size=3)*0.01)
            out = a.update(T, t)
            cm, Rm = _split(T)
            if c is None:
                c, R, pt = cm, Rm, t
            else:
                w = alpha(0.7, t-pt)
                c, R, pt = c+w*(cm-c), exp(w*log(Rm @ R.T)) @ R, t
            self.assertLess(np.abs(_split(out)[0]-c).max(), 1e-12)
            self.assertLess(np.abs(log(_split(out)[1] @ R.T)).max(), 1e-12)

    def test_reset_returns_the_measurement(self):
        T0, T1 = pose([0, 0, 1]), pose([0.5, 0.1, 1.2], (0.2, 0, 0))
        for f in (PoseSmoother(P()), OneEuroPose(P()), KalmanPose(P())):
            f.update(T0, 0.0)
            f.update(T0, 0.1)
            f.reset()
            self.assertLess(np.abs(f.update(T1, 5.0)-T1).max(), 1e-12)

    def test_gate_rejects_a_jump_then_resets_after_two_frames(self):
        f, T = KalmanPose(P(q=1e-4)), pose([0, 0, 1.5])
        for k in range(30):
            f.update(T, 0.1*k)
        jump = pose([0.2, 0, 1.5])
        out = f.update(jump, 3.0)
        self.assertTrue(f.gated)
        self.assertLess(move(out, T)[0], 0.05)                         # prediction only, not the jump
        out = f.update(jump, 3.1)
        self.assertTrue(f.gated)                                        # second skipped frame -> reset to the measurement
        self.assertLess(np.abs(out-jump).max(), 1e-12)
        out = f.update(jump, 3.2)
        self.assertFalse(f.gated)

    def test_covariance_conversion_is_symmetric_positive(self):
        T = pose([0.3, -0.2, 1.5], (0.1, 0.2, -0.1))
        S = covariance_wc(T, np.diag([1e-5]*3+[1e-6]*3))
        self.assertLess(np.abs(S-S.T).max(), 1e-12)
        self.assertGreater(np.linalg.eigvalsh(S).min(), 0)

    def test_make_smoother_modes(self):
        base = load_config(CFG).filter
        from dataclasses import replace
        self.assertIsInstance(make_smoother(base), PoseSmoother)
        self.assertIsInstance(make_smoother(replace(base, mode="one_euro")), OneEuroPose)
        self.assertTrue(make_smoother(replace(base, mode="kalman_cov")).use_covariance)

    @unittest.skipUnless(RUN.exists(), "static run not available")
    def test_offline_ema_reproduces_logged_render_pose(self):
        cfg = load_config(RUN/"config.json")
        f, prev_valid, prev_t, n = PoseSmoother(cfg.filter), False, None, 0
        for line in open(RUN/"frames.jsonl"):
            r = json.loads(line)
            if n >= 100:
                break
            t, valid = r["source_time_s"], r["raw_pose"]["valid"]
            if valid:
                if not prev_valid or (prev_t is not None and t-prev_t > cfg.pose.max_prediction_gap_s):
                    f.reset()
                out = f.update(np.array(r["raw_pose"]["T_CW"]), t, None)
                self.assertLess(np.abs(out-np.array(r["render_T_CW"])).max(), 1e-6)
                n += 1
            prev_valid, prev_t = valid, t
        self.assertEqual(n, 100)


if __name__ == "__main__":
    unittest.main()
