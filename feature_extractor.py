import numpy as np
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional
from pose_estimator import PoseResult, Landmark


@dataclass
class RunningFeatures:
    knee_angle_left:  Optional[float]  # 고관절-무릎-발목 (좌), 관절 가림 시 None
    knee_angle_right: Optional[float]  # 고관절-무릎-발목 (우)
    hip_angle_left:   Optional[float]  # 어깨-고관절-무릎 (좌)
    hip_angle_right:  Optional[float]  # 어깨-고관절-무릎 (우)
    torso_lean:       Optional[float]  # 상체 기울기 (수직 기준)
    arm_swing_left:   Optional[float]  # 어깨-팔꿈치-손목 (좌)
    arm_swing_right:  Optional[float]  # 어깨-팔꿈치-손목 (우)
    cadence: float                     # 분당 보행 횟수 (항상 계산)
    timestamp: float

    def has_any(self) -> bool:
        """계산된 피처가 하나라도 있으면 True."""
        return any(v is not None for v in [
            self.knee_angle_left, self.knee_angle_right,
            self.hip_angle_left,  self.hip_angle_right,
            self.torso_lean, self.arm_swing_left, self.arm_swing_right,
        ])


# 피처별로 필요한 관절 정의
FEATURE_JOINTS: dict[str, list[str]] = {
    "knee_angle_left":  ["left_hip",   "left_knee",  "left_ankle"],
    "knee_angle_right": ["right_hip",  "right_knee", "right_ankle"],
    "hip_angle_left":   ["left_shoulder",  "left_hip",  "left_knee"],
    "hip_angle_right":  ["right_shoulder", "right_hip", "right_knee"],
    "torso_lean":       ["left_shoulder", "right_shoulder", "left_hip", "right_hip"],
    "arm_swing_left":   ["left_shoulder",  "left_elbow",  "left_wrist"],
    "arm_swing_right":  ["right_shoulder", "right_elbow", "right_wrist"],
}

VISIBILITY_THRESHOLD = 0.25  # 트레드밀 손잡이 가림 허용 (0.5 → 0.25)


def _visible(lm: dict, *joints: str) -> bool:
    return all(lm[j].visibility >= VISIBILITY_THRESHOLD for j in joints)


def _angle_3points(a: Landmark, b: Landmark, c: Landmark) -> float:
    """b를 꼭짓점으로 하는 a-b-c 세 점의 3D 각도 (도 단위).
    월드 좌표 기반이므로 카메라 각도 변화에 안정적."""
    ba = np.array([a.x - b.x, a.y - b.y, a.z - b.z])
    bc = np.array([c.x - b.x, c.y - b.y, c.z - b.z])
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return 180.0
    cos_angle = np.clip(np.dot(ba, bc) / (norm_ba * norm_bc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def _torso_lean_angle(shoulder: Landmark, hip: Landmark) -> float:
    """힙→어깨 3D 벡터와 수직축 사이의 각도.
    MediaPipe world 좌표계에서 Y축이 아래 방향이므로 수직 상방은 (0, -1, 0)."""
    vec = np.array([shoulder.x - hip.x, shoulder.y - hip.y, shoulder.z - hip.z])
    norm = np.linalg.norm(vec)
    if norm < 1e-6:
        return 0.0
    vec /= norm
    return float(np.degrees(np.arccos(np.clip(-vec[1], -1.0, 1.0))))


class CadenceTracker:
    """발목 Y좌표의 주기적 변화로 케이던스 추정."""

    def __init__(self, window_sec: float = 5.0):
        self._window_sec = window_sec
        self._history: deque[tuple[float, float]] = deque()
        self._last_peak_times: deque[float] = deque()
        self._prev_y: Optional[float] = None
        self._prev_dy: float = 0.0

    def update(self, ankle_y: float, timestamp: float) -> float:
        self._history.append((timestamp, ankle_y))
        while self._history and self._history[0][0] < timestamp - self._window_sec:
            self._history.popleft()

        if self._prev_y is not None:
            dy = ankle_y - self._prev_y
            if self._prev_dy < 0 and dy >= 0:
                self._last_peak_times.append(timestamp)
                while self._last_peak_times and self._last_peak_times[0] < timestamp - self._window_sec:
                    self._last_peak_times.popleft()

        self._prev_dy = ankle_y - (self._prev_y or ankle_y)
        self._prev_y = ankle_y

        n = len(self._last_peak_times)
        if n < 2:
            return 0.0
        elapsed = self._last_peak_times[-1] - self._last_peak_times[0]
        return (n - 1) / elapsed * 60.0 if elapsed >= 0.5 else 0.0


class FeatureExtractor:
    def __init__(self):
        self._left_cadence  = CadenceTracker()
        self._right_cadence = CadenceTracker()

    def extract(self, pose: PoseResult, video_ts: Optional[float] = None) -> Optional[RunningFeatures]:
        lm  = pose.landmarks        # 이미지 좌표: 가시성 체크 + 케이던스용
        wlm = pose.world_landmarks  # 3D 월드 좌표: 각도 계산용
        t = video_ts if video_ts is not None else time.time()

        # ── 피처별 독립 계산 (가시성=이미지, 각도=월드 3D) ────────
        knee_l = (
            _angle_3points(wlm["left_hip"], wlm["left_knee"], wlm["left_ankle"])
            if _visible(lm, "left_hip", "left_knee", "left_ankle") else None
        )
        knee_r = (
            _angle_3points(wlm["right_hip"], wlm["right_knee"], wlm["right_ankle"])
            if _visible(lm, "right_hip", "right_knee", "right_ankle") else None
        )
        hip_l = (
            _angle_3points(wlm["left_shoulder"], wlm["left_hip"], wlm["left_knee"])
            if _visible(lm, "left_shoulder", "left_hip", "left_knee") else None
        )
        hip_r = (
            _angle_3points(wlm["right_shoulder"], wlm["right_hip"], wlm["right_knee"])
            if _visible(lm, "right_shoulder", "right_hip", "right_knee") else None
        )

        if _visible(lm, "left_shoulder", "right_shoulder", "left_hip", "right_hip"):
            mid_sh = Landmark(
                x=(wlm["left_shoulder"].x + wlm["right_shoulder"].x) / 2,
                y=(wlm["left_shoulder"].y + wlm["right_shoulder"].y) / 2,
                z=(wlm["left_shoulder"].z + wlm["right_shoulder"].z) / 2,
                visibility=1,
            )
            mid_hp = Landmark(
                x=(wlm["left_hip"].x + wlm["right_hip"].x) / 2,
                y=(wlm["left_hip"].y + wlm["right_hip"].y) / 2,
                z=(wlm["left_hip"].z + wlm["right_hip"].z) / 2,
                visibility=1,
            )
            torso = _torso_lean_angle(mid_sh, mid_hp)
        else:
            torso = None

        arm_l = (
            _angle_3points(wlm["left_shoulder"], wlm["left_elbow"], wlm["left_wrist"])
            if _visible(lm, "left_shoulder", "left_elbow", "left_wrist") else None
        )
        arm_r = (
            _angle_3points(wlm["right_shoulder"], wlm["right_elbow"], wlm["right_wrist"])
            if _visible(lm, "right_shoulder", "right_elbow", "right_wrist") else None
        )

        # 케이던스: 이미지 좌표 발목 Y 사용 (진동 패턴 감지)
        cad_l = (self._left_cadence.update(lm["left_ankle"].y, t)
                 if _visible(lm, "left_ankle") else 0.0)
        cad_r = (self._right_cadence.update(lm["right_ankle"].y, t)
                 if _visible(lm, "right_ankle") else 0.0)
        valid_cads = [c for c in [cad_l, cad_r] if c > 0]
        cadence = sum(valid_cads) / len(valid_cads) * 2 if valid_cads else 0.0

        feat = RunningFeatures(
            knee_angle_left=knee_l,
            knee_angle_right=knee_r,
            hip_angle_left=hip_l,
            hip_angle_right=hip_r,
            torso_lean=torso,
            arm_swing_left=arm_l,
            arm_swing_right=arm_r,
            cadence=cadence,
            timestamp=t,
        )
        # 계산된 피처가 하나도 없으면 None 반환
        return feat if feat.has_any() else None

    # AE 입력용 벡터: None은 정상 범위 중앙값으로 대체 (imputation)
    IMPUTE_DEFAULTS = np.array([160., 160., 170., 170., 5., 90., 90., 170.], dtype=np.float32)

    def to_vector(self, features: RunningFeatures) -> np.ndarray:
        vals = [
            features.knee_angle_left,
            features.knee_angle_right,
            features.hip_angle_left,
            features.hip_angle_right,
            features.torso_lean,
            features.arm_swing_left,
            features.arm_swing_right,
            features.cadence,
        ]
        return np.array(
            [v if v is not None else self.IMPUTE_DEFAULTS[i] for i, v in enumerate(vals)],
            dtype=np.float32,
        )
