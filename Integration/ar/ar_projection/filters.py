"""Step 51 pose filters: 1 euro and constant-velocity Kalman, same interface as rendering.PoseSmoother.

update(T_CW, t=None, covariance=None) -> T_CW and reset(). All filters work on the camera pose in the world,
T_WC = inverse(T_CW): position c and rotation R. Rotation errors are rotation vectors, e = Log(R_m R^T) and R <- Exp(a) R.
The caller resets the filter on the first valid pose after an invalid frame and after a time gap (pipeline.py)."""
import cv2
import numpy as np
from .config import DEFAULT_SIGMA
from .geometry import inverse
from .rendering import PoseSmoother

GATE_CHI2_6 = 22.46          # chi-square, 6 dof, p = 0.999


def exp(v):
    return cv2.Rodrigues(np.asarray(v, np.float64).reshape(3, 1))[0]


def log(R):
    return cv2.Rodrigues(np.asarray(R, np.float64))[0].ravel()


def alpha(cutoff_hz, dt):
    return 1.0/(1.0+1.0/(2*np.pi*cutoff_hz)/dt)


def _split(T_CW):
    W = inverse(T_CW)
    return W[:3, 3].copy(), W[:3, :3].copy()


def _join(c, R):
    W = np.eye(4)
    W[:3, :3], W[:3, 3] = R, c
    return inverse(W)


def covariance_wc(T_CW, cov, h=1e-3):   # h well above 1e-5 rad: cv2.Rodrigues returns 0 for smaller rotations
    """6x6 covariance of the logged left perturbation of T_CW ([dt, dr], T' = [Exp(dr) dt; 0 1] T_CW) expressed as
    (camera position in world, rotation error Log(R' R^T) of R_WC): Sigma' = J Sigma J^T, central differences."""
    c0, R0 = _split(T_CW)
    J = np.zeros((6, 6))
    for i in range(6):
        out = []
        for s in (h, -h):
            d = np.zeros(6)
            d[i] = s
            P = np.eye(4)
            P[:3, :3], P[:3, 3] = exp(d[3:]), d[:3]
            c, R = _split(P @ T_CW)
            out.append(np.r_[c-c0, log(R @ R0.T)])
        J[:, i] = (out[0]-out[1])/(2*h)
    return J @ np.asarray(cov, np.float64) @ J.T


class OneEuroPose:
    """Casiez et al., CHI 2012, on the 6-DoF pose with one shared adaptive cutoff (speed in m/s: translation plus
    scene_distance_m x rotation). beta = 0 gives a dt-aware EMA with weight alpha(min_cutoff_hz, dt)."""

    def __init__(self, config):
        self.min_cutoff, self.beta, self.d_cutoff, self.d = config.min_cutoff_hz, config.beta, config.d_cutoff_hz, config.scene_distance_m
        self.reset()

    def reset(self):
        self.c = self.R = self.t = None
        self.speed = 0.0

    def update(self, T, t=None, covariance=None):
        cm, Rm = _split(T)
        if self.c is None or t is None or t <= self.t:
            self.c, self.R, self.t, self.speed = cm, Rm, t, 0.0
            return T.copy()
        dt = t-self.t
        e = log(Rm @ self.R.T)
        raw_speed = (np.linalg.norm(cm-self.c)+self.d*np.linalg.norm(e))/dt
        self.speed += alpha(self.d_cutoff, dt)*(raw_speed-self.speed)
        a = alpha(self.min_cutoff+self.beta*self.speed, dt)
        self.c = self.c+a*(cm-self.c)
        self.R = exp(a*e) @ self.R
        self.t = t
        return _join(self.c, self.R)


class KalmanPose:
    """Constant-velocity Kalman filter, state [c, theta, v, omega] (theta = rotation error about the current R, folded into R
    after each update). Process noise: continuous white-noise acceleration q (m^2/s^3; rotation q/d^2). Measurement noise:
    diag(sigma^2) from config.measurement_sigma, or with use_covariance r_scale x the logged pose covariance converted by
    covariance_wc (frames without one fall back to the fixed noise). Innovation gate: chi-square 6 dof; a gated frame is a
    prediction only and the second gated frame in a row resets to the measurement."""

    def __init__(self, config, use_covariance=False):
        self.q, self.d, self.r_scale, self.gate = config.q, config.scene_distance_m, config.r_scale, config.gate_chi2
        self.Rfix = np.diag(np.square(np.asarray(config.measurement_sigma or DEFAULT_SIGMA, np.float64)))
        self.use_covariance = use_covariance
        self.reset()

    def reset(self):
        self.x = self.R = self.t = self.P = None
        self.skipped = 0
        self.gated = False                  # last update was a gated prediction
        self.nis = None                     # d^2 of the last update

    def _start(self, c, R, t, Rm):
        self.x = np.zeros(12)
        self.x[:3] = c
        self.R, self.t = R, t
        self.P = np.zeros((12, 12))
        self.P[:6, :6] = Rm
        self.P[6:, 6:] = np.eye(6)
        self.skipped = 0

    def update(self, T, t=None, covariance=None):
        cm, Rm_ = _split(T)
        Rn = self.Rfix
        if self.use_covariance and covariance is not None:
            Rn = self.r_scale*covariance_wc(T, covariance)
        self.gated, self.nis = False, None
        if self.x is None or t is None or t <= self.t:
            self._start(cm, Rm_, t, Rn)
            return T.copy()
        dt = t-self.t
        F = np.eye(12)
        F[:6, 6:] = dt*np.eye(6)
        Qd = self.q*np.diag([1, 1, 1, 1/self.d**2, 1/self.d**2, 1/self.d**2])
        Q = np.block([[dt**3/3*Qd, dt**2/2*Qd], [dt**2/2*Qd, dt*Qd]])
        self.x = F @ self.x
        self.R = exp(self.x[9:12]*dt) @ self.R
        self.x[3:6] = 0.0
        self.P = F @ self.P @ F.T+Q
        y = np.r_[cm-self.x[:3], log(Rm_ @ self.R.T)]
        S = self.P[:6, :6]+Rn
        d2 = float(y @ np.linalg.solve(S, y))
        self.t = t
        if d2 > self.gate:
            self.skipped += 1
            self.gated = True
            if self.skipped >= 2:
                self._start(cm, Rm_, t, Rn)
                return T.copy()
            return _join(self.x[:3], self.R)
        self.skipped = 0
        self.nis = d2
        K = np.linalg.solve(S, self.P[:6, :]).T
        self.x = self.x+K @ y
        self.P = self.P-K @ self.P[:6, :]
        self.P = (self.P+self.P.T)/2
        self.R = exp(self.x[3:6]) @ self.R
        self.x[3:6] = 0.0
        return _join(self.x[:3], self.R)


def make_smoother(config):
    """The pose smoother named by filter.mode; a disabled filter is the pass-through PoseSmoother."""
    if not config.enabled or config.mode == "ema":
        return PoseSmoother(config)
    if config.mode == "one_euro":
        return OneEuroPose(config)
    return KalmanPose(config, use_covariance=config.mode == "kalman_cov")
