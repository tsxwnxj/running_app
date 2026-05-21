"""자세 이상 탐지 모듈.

Rule-based + Autoencoder 병행.
Rule-based는 보행 단계(GaitPhase)별 정상 범위를 적용.
"""

import os
import sys
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, os.path.dirname(__file__))
from feature_extractor import RunningFeatures
from gait_phase import GaitPhase

MODEL_PATH      = os.path.join(os.path.dirname(__file__), "model", "running_ae.pt")
THRESHOLD_PATH  = os.path.join(os.path.dirname(__file__), "model", "threshold.txt")

def _load_threshold(default: float = 0.5) -> float:
    try:
        return float(open(THRESHOLD_PATH).read().strip())
    except Exception:
        return default


# ── 정상 범위 ────────────────────────────────────────────────────
#
# knee_angle  = ∠(Hip-Knee-Ankle)
#   전방 무릎(해당 단계에서 앞으로 나온 쪽): 155-170°
#   후방 무릎(해당 단계에서 뒤로 간 쪽):    50-90°
#
# hip_angle   = ∠(Shoulder-Hip-Knee): 모든 단계 135-155°
# torso_lean  = shoulder→hip 벡터와 수직축 사이 각도: 5-10°
# arm_swing   = ∠(Shoulder-Elbow-Wrist): 모든 단계 50-100°

KNEE_FORWARD_RANGE  = (130.0, 167.0)  # 실측 ±5° 조정
KNEE_BACKWARD_RANGE = (60.0,  160.0)

NORMAL_RANGES: dict[str, tuple[float, float]] = {
    "knee_angle": (60.0,  167.0),
    "hip_angle":  (135.0, 171.0),
    "torso_lean": (7.0,   13.0),
    "arm_swing":  (75.0,  125.0),
}

AUTOENCODER_THRESHOLD = _load_threshold(default=0.5)

# AE 입력 대체값 (None 피처용)
_IMPUTE = np.array([160., 160., 170., 170., 5., 90., 90., 0.], dtype=np.float32)


@dataclass
class AnomalyResult:
    is_anomaly: bool
    ae_error: float
    rule_violations: dict[str, float] = field(default_factory=dict)


def _get_range(
    feature: str,
    phase: Optional[GaitPhase],
    side: Optional[str] = None,
) -> tuple[float, float]:
    """정상 범위 반환. knee_angle은 phase+side 조합으로 전방/후방 구분."""
    if feature == "knee_angle" and phase in (
        GaitPhase.LEFT_KNEE_FORWARD, GaitPhase.RIGHT_KNEE_FORWARD
    ):
        is_forward = (
            (phase == GaitPhase.LEFT_KNEE_FORWARD  and side == "left") or
            (phase == GaitPhase.RIGHT_KNEE_FORWARD and side == "right")
        )
        return KNEE_FORWARD_RANGE if is_forward else KNEE_BACKWARD_RANGE
    return NORMAL_RANGES.get(feature, (0.0, 360.0))


class AnomalyDetector:
    def __init__(self):
        self._model = None
        self._normalizer = None
        self._load_or_train()

    def _load_or_train(self):
        from model.autoencoder import (
            RunningAutoencoder, FeatureNormalizer,
            generate_normal_running_data, train_autoencoder,
            save_model, load_model,
        )
        if os.path.exists(MODEL_PATH):
            self._model, self._normalizer = load_model(MODEL_PATH)
        else:
            print("[AnomalyDetector] 모델 학습 중... (최초 1회)")
            data = generate_normal_running_data(n=5000)
            self._normalizer = FeatureNormalizer()
            self._model = RunningAutoencoder()
            train_autoencoder(self._model, self._normalizer, data, epochs=150)
            save_model(self._model, self._normalizer, MODEL_PATH)
            print(f"[AnomalyDetector] 모델 저장 완료: {MODEL_PATH}")

    def detect(
        self,
        features: RunningFeatures,
        phase: Optional[GaitPhase] = None,
    ) -> AnomalyResult:
        violations = self._rule_check(features, phase)

        vec = self._to_vector(features)
        norm_vec = (vec - self._normalizer.mean) / self._normalizer.std
        t = torch.from_numpy(norm_vec).unsqueeze(0)
        ae_error = float(self._model.reconstruction_error(t).item())

        # 이상감지율(anomaly_rate)은 AE 기준만 사용
        # 규칙 기반 violations는 피드백 메시지 생성에만 활용
        is_anomaly = ae_error > AUTOENCODER_THRESHOLD
        return AnomalyResult(
            is_anomaly=is_anomaly,
            ae_error=ae_error,
            rule_violations=violations,
        )

    def _rule_check(
        self,
        f: RunningFeatures,
        phase: Optional[GaitPhase] = None,
    ) -> dict[str, float]:
        violations: dict[str, float] = {}

        def check(feat: str, value: Optional[float], lo: float, hi: float):
            if value is None:
                return
            deviation = max(0.0, lo - value, value - hi)
            if deviation >= 5.0:
                violations[feat] = max(violations.get(feat, 0.0), deviation)

        kl, kr = _get_range("knee_angle", phase, "left"), _get_range("knee_angle", phase, "right")
        check("knee_angle", f.knee_angle_left,  *kl)
        check("knee_angle", f.knee_angle_right, *kr)
        check("hip_angle",  f.hip_angle_left,   *NORMAL_RANGES["hip_angle"])
        check("hip_angle",  f.hip_angle_right,  *NORMAL_RANGES["hip_angle"])
        check("torso_lean", f.torso_lean,        *NORMAL_RANGES["torso_lean"])
        check("arm_swing",  f.arm_swing_left,    *NORMAL_RANGES["arm_swing"])
        check("arm_swing",  f.arm_swing_right,   *NORMAL_RANGES["arm_swing"])
        return violations

    def get_per_feature_score(
        self,
        features: RunningFeatures,
        phase: Optional[GaitPhase] = None,
    ) -> dict[str, float]:
        """각 피처 정상 범위 점수 (0~1, 1=정상, -1=측정 불가)."""
        scores: dict[str, float] = {}

        def score(value: Optional[float], lo: float, hi: float) -> float:
            if value is None:
                return -1.0
            mid = (lo + hi) / 2
            half = (hi - lo) / 2
            dev = abs(value - mid)
            return max(0.0, 1.0 - (dev - half) / half) if dev > half else 1.0

        kl_range = _get_range("knee_angle", phase, "left")
        kr_range = _get_range("knee_angle", phase, "right")
        sl = score(features.knee_angle_left,  *kl_range)
        sr = score(features.knee_angle_right, *kr_range)
        valid = [s for s in [sl, sr] if s >= 0]
        scores["knee_angle"] = min(valid) if valid else -1.0

        hip_r = NORMAL_RANGES["hip_angle"]
        sl = score(features.hip_angle_left,  *hip_r)
        sr = score(features.hip_angle_right, *hip_r)
        valid = [s for s in [sl, sr] if s >= 0]
        scores["hip_angle"] = min(valid) if valid else -1.0

        scores["torso_lean"] = score(features.torso_lean, *NORMAL_RANGES["torso_lean"])

        arm_r = NORMAL_RANGES["arm_swing"]
        sl = score(features.arm_swing_left,  *arm_r)
        sr = score(features.arm_swing_right, *arm_r)
        valid = [s for s in [sl, sr] if s >= 0]
        scores["arm_swing"] = min(valid) if valid else -1.0

        return scores

    @staticmethod
    def _to_vector(features: RunningFeatures) -> np.ndarray:
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
            [v if v is not None else _IMPUTE[i] for i, v in enumerate(vals)],
            dtype=np.float32,
        )
