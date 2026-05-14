"""영상 파일을 프레임별로 분석하여 스켈레톤 오버레이 영상과 메트릭을 반환.

처리 순서:
  Pass 1  – 포즈 추정 + 피처 수집 (이상 감지 제외)
  Smooth  – 전체 시퀀스 Gaussian 스무딩
  Phase   – 보행 단계 감지
  Detect  – 단계별 이상 탐지
  Pass 2  – 스무딩 랜드마크로 스켈레톤 오버레이 렌더링
"""

import os
import sys
import time
import numpy as np
import cv2
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict

sys.path.insert(0, str(Path(__file__).parent))

from pose_estimator import PoseEstimator, PoseResult, Landmark
from feature_extractor import FeatureExtractor, RunningFeatures
from anomaly_detector import AnomalyDetector, NORMAL_RANGES
from feedback_generator import FeedbackGenerator
from gait_phase import GaitPhaseDetector, GaitPhase, PHASE_LABELS_KO

try:
    from PIL import ImageFont, ImageDraw, Image as PILImage
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/Library/Fonts/NanumGothic.ttf",
]

# ── 색상 (BGR) ─────────────────────────────────────────────────
C_GOOD   = (80, 200, 80)
C_WARN   = (40, 180, 255)
C_ERROR  = (60, 60, 230)
C_WHITE  = (240, 240, 240)
C_DARK   = (20, 20, 30)
C_ACCENT = (255, 160, 40)


def _find_font() -> Optional[str]:
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            return p
    return None


def _pil_text(img: np.ndarray, text: str, pos, size: int, color) -> np.ndarray:
    font_path = _find_font()
    if not _HAS_PIL or not font_path:
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, size / 28,
                    (color[2], color[1], color[0]) if len(color) == 3 else color, 1, cv2.LINE_AA)
        return img
    pil = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font = ImageFont.truetype(font_path, size)
    draw.text(pos, text, font=font, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def _level_color(level: str):
    return {"good": C_GOOD, "warning": C_WARN, "error": C_ERROR}.get(level, C_WHITE)


def _score_color(score: float):
    if score >= 0.85:
        return C_GOOD
    if score >= 0.55:
        return C_WARN
    return C_ERROR


# ── 스켈레톤 연결 ──────────────────────────────────────────────
SKELETON = [
    ("left_shoulder",  "right_shoulder"),
    ("left_shoulder",  "left_elbow"),
    ("left_elbow",     "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow",    "right_wrist"),
    ("left_shoulder",  "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip",       "right_hip"),
    ("left_hip",       "left_knee"),
    ("left_knee",      "left_ankle"),
    ("right_hip",      "right_knee"),
    ("right_knee",     "right_ankle"),
]

JOINT_FEATURE = {
    "left_knee":      "knee_angle",
    "right_knee":     "knee_angle",
    "left_hip":       "hip_angle",
    "right_hip":      "hip_angle",
    "left_shoulder":  "arm_swing",
    "right_shoulder": "arm_swing",
    "left_elbow":     "arm_swing",
    "right_elbow":    "arm_swing",
}


def _draw_skeleton(
    frame: np.ndarray,
    lm: dict,
    scores: dict[str, float],
) -> np.ndarray:
    out = frame.copy()

    def pt(name):
        l = lm.get(name)
        return (int(l.x), int(l.y)) if l else None

    def jcol(name):
        feat = JOINT_FEATURE.get(name)
        s = scores.get(feat, 1.0) if feat else 1.0
        return _score_color(s)

    for a, b in SKELETON:
        pa, pb = pt(a), pt(b)
        if pa and pb:
            col = _score_color(
                min(scores.get(JOINT_FEATURE.get(a, ""), 1.0),
                    scores.get(JOINT_FEATURE.get(b, ""), 1.0))
            )
            cv2.line(out, pa, pb, col, 3, cv2.LINE_AA)

    for name in lm:
        p = pt(name)
        if p:
            cv2.circle(out, p, 6, (255, 255, 255), -1)
            cv2.circle(out, p, 6, jcol(name), 2)

    return out


def _draw_hud(
    frame: np.ndarray,
    features: RunningFeatures,
    scores: dict[str, float],
    is_anomaly: bool,
    phase: GaitPhase,
) -> np.ndarray:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 52), C_DARK, -1)
    frame = cv2.addWeighted(frame, 0.3, overlay, 0.7, 0)

    status_text  = "자세 이상 감지" if is_anomaly else "정상 자세"
    status_color = C_ERROR if is_anomaly else C_GOOD
    phase_ko     = PHASE_LABELS_KO.get(phase, "")

    frame = _pil_text(frame, "러닝 자세 분석", (12, 8), 18, C_ACCENT)
    frame = _pil_text(frame, status_text, (w - 220, 8), 18, status_color)
    if phase_ko:
        frame = _pil_text(frame, phase_ko, (12, 30), 13, C_WHITE)

    if is_anomaly:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), C_ERROR, 4)

    def fmt0(val):
        return f"{val:.0f}°" if val is not None else "--°"

    def fmt1(val):
        return f"{val:.1f}°" if val is not None else "--°"

    panel_lines = [
        f"무릎(좌): {fmt0(features.knee_angle_left)}",
        f"무릎(우): {fmt0(features.knee_angle_right)}",
        f"고관절(좌): {fmt0(features.hip_angle_left)}",
        f"고관절(우): {fmt0(features.hip_angle_right)}",
        f"상체기울기: {fmt1(features.torso_lean)}",
        f"팔스윙(좌): {fmt0(features.arm_swing_left)}",
        f"팔스윙(우): {fmt0(features.arm_swing_right)}",
    ]

    line_h  = 22
    panel_h = len(panel_lines) * line_h + 16
    panel_w = 200
    px, py  = w - panel_w - 10, h - panel_h - 10

    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (px - 4, py - 4), (w - 6, h - 6), C_DARK, -1)
    frame = cv2.addWeighted(frame, 0.35, overlay2, 0.65, 0)

    for i, line in enumerate(panel_lines):
        frame = _pil_text(frame, line, (px, py + i * line_h), 14, C_WHITE)

    return frame


@dataclass
class FrameMetric:
    timestamp: float
    knee_angle_left:  Optional[float]
    knee_angle_right: Optional[float]
    hip_angle_left:   Optional[float]
    hip_angle_right:  Optional[float]
    torso_lean:       Optional[float]
    arm_swing_left:   Optional[float]
    arm_swing_right:  Optional[float]
    cadence:          float
    phase:            str   # GaitPhase.value
    is_anomaly:       bool


def _stats(vals: list) -> Optional[dict]:
    valid = [v for v in vals if v is not None]
    if not valid:
        return None
    arr = np.array(valid, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std()),
            "min": float(arr.min()), "max": float(arr.max())}


def compute_aggregate(frames: list[FrameMetric]) -> dict:
    if not frames:
        return {}
    anomaly_rate = sum(1 for f in frames if f.is_anomaly) / len(frames)
    return {
        "frame_count":      len(frames),
        "anomaly_rate":     round(anomaly_rate, 3),
        "knee_angle_left":  _stats([f.knee_angle_left  for f in frames]),
        "knee_angle_right": _stats([f.knee_angle_right for f in frames]),
        "hip_angle_left":   _stats([f.hip_angle_left   for f in frames]),
        "hip_angle_right":  _stats([f.hip_angle_right  for f in frames]),
        "torso_lean":       _stats([f.torso_lean        for f in frames]),
        "arm_swing_left":   _stats([f.arm_swing_left   for f in frames]),
        "arm_swing_right":  _stats([f.arm_swing_right  for f in frames]),
    }


def _ema_landmarks(
    raw_seq: list[Optional[dict]],
    alpha: float = 0.5,
) -> list[Optional[dict]]:
    """EMA 스무딩. alpha=현재 프레임 가중치 (0~1, 클수록 덜 스무딩)."""
    result: list[Optional[dict]] = []
    prev: Optional[dict] = None

    for lm in raw_seq:
        if lm is None:
            result.append(prev)
            continue
        if prev is None:
            result.append(lm)
            prev = lm
            continue
        smoothed = {
            key: Landmark(
                x=alpha * lm[key].x + (1 - alpha) * prev[key].x,
                y=alpha * lm[key].y + (1 - alpha) * prev[key].y,
                z=alpha * lm[key].z + (1 - alpha) * prev[key].z,
                visibility=lm[key].visibility,
            )
            for key in lm
        }
        result.append(smoothed)
        prev = smoothed

    return result


def process_video(
    input_path: str,
    output_path: str,
    view_angle: str = "side",
    progress_cb=None,
) -> dict:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"영상을 열 수 없습니다: {input_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    estimator    = PoseEstimator()
    extractor    = FeatureExtractor()
    detector     = AnomalyDetector()
    feedback_gen = FeedbackGenerator()

    # ── Pass 1: 포즈 추정 + 피처 수집 ──────────────────────────────
    raw_landmarks: list[Optional[dict]]           = []
    raw_features:  list[Optional[RunningFeatures]] = []
    frames_buf:    list[np.ndarray]                = []
    raw_ts:        list[float]                     = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames_buf.append(frame)
        ts = frame_idx / fps
        raw_ts.append(ts)

        pose = estimator.process(frame, int(ts * 1000))
        if pose:
            raw_landmarks.append(pose.landmarks)
            raw_features.append(extractor.extract(pose, video_ts=ts))
        else:
            raw_landmarks.append(None)
            raw_features.append(None)

        frame_idx += 1
        if progress_cb:
            progress_cb(frame_idx, total * 2)

    cap.release()
    estimator.close()

    # ── 스무딩 ─────────────────────────────────────────────────────
    smoothed_landmarks = _ema_landmarks(raw_landmarks, alpha=0.5)

    # ── 보행 단계 감지 ──────────────────────────────────────────────
    phase_detector = GaitPhaseDetector(fps=fps)
    phases = phase_detector.detect(smoothed_landmarks)

    # ── 단계별 이상 탐지 ────────────────────────────────────────────
    frame_scores:  list[dict]        = []
    frame_metrics: list[FrameMetric] = []

    for i, (features, phase) in enumerate(zip(raw_features, phases)):
        if features:
            anomaly = detector.detect(features, phase=phase)
            scores  = detector.get_per_feature_score(features, phase=phase)
            frame_scores.append(scores)
            frame_metrics.append(FrameMetric(
                timestamp=round(raw_ts[i], 3),
                knee_angle_left=features.knee_angle_left,
                knee_angle_right=features.knee_angle_right,
                hip_angle_left=features.hip_angle_left,
                hip_angle_right=features.hip_angle_right,
                torso_lean=features.torso_lean,
                arm_swing_left=features.arm_swing_left,
                arm_swing_right=features.arm_swing_right,
                cadence=features.cadence,
                phase=phase.value,
                is_anomaly=anomaly.is_anomaly,
            ))
        else:
            frame_scores.append({})

    # ── 집계 메트릭 ─────────────────────────────────────────────────
    metrics = compute_aggregate(frame_metrics)

    def _mean(key):
        s = metrics.get(key)
        return s["mean"] if s else None

    # 렌더링용 대표 피처 (전체 평균)
    import time as _t
    from feature_extractor import RunningFeatures as RF

    avg_feat = RF(
        knee_angle_left=_mean("knee_angle_left"),
        knee_angle_right=_mean("knee_angle_right"),
        hip_angle_left=_mean("hip_angle_left"),
        hip_angle_right=_mean("hip_angle_right"),
        torso_lean=_mean("torso_lean"),
        arm_swing_left=_mean("arm_swing_left"),
        arm_swing_right=_mean("arm_swing_right"),
        cadence=0.0,
        timestamp=_t.time(),
    ) if frame_metrics else None

    last_anomaly = (metrics.get("anomaly_rate", 0) > 0.2) if metrics else False

    # ── Pass 2: 렌더링 ──────────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    score_iter  = iter(frame_scores)
    phase_iter  = iter(phases)
    last_scores: dict = {}
    last_phase  = GaitPhase.UNKNOWN

    for i, (frame, lm) in enumerate(zip(frames_buf, smoothed_landmarks)):
        scores = next(score_iter, {})
        phase  = next(phase_iter, GaitPhase.UNKNOWN)
        if scores:
            last_scores = scores
        if phase != GaitPhase.UNKNOWN:
            last_phase = phase

        annotated = _draw_skeleton(frame, lm, last_scores) if lm else frame.copy()

        if avg_feat:
            annotated = _draw_hud(
                annotated, avg_feat, last_scores,
                last_anomaly, last_phase,
            )

        out_writer.write(annotated)
        if progress_cb:
            progress_cb(total + i + 1, total * 2)

    out_writer.release()

    # ── 단계별 피드백 생성 ──────────────────────────────────────────
    feedback_msgs = []
    if frame_metrics:
        # 단계별 평균 피처 계산
        phase_frames: dict[str, list[FrameMetric]] = {}
        for fm in frame_metrics:
            phase_frames.setdefault(fm.phase, []).append(fm)

        def phase_mean(fms: list[FrameMetric], key: str) -> Optional[float]:
            vals = [getattr(fm, key) for fm in fms if getattr(fm, key) is not None]
            return float(np.mean(vals)) if vals else None

        from anomaly_detector import AnomalyResult

        seen: set[str] = set()
        for phase_str, pfms in sorted(phase_frames.items()):
            try:
                phase = GaitPhase(phase_str)
            except ValueError:
                continue

            pf = RF(
                knee_angle_left=phase_mean(pfms, "knee_angle_left"),
                knee_angle_right=phase_mean(pfms, "knee_angle_right"),
                hip_angle_left=phase_mean(pfms, "hip_angle_left"),
                hip_angle_right=phase_mean(pfms, "hip_angle_right"),
                torso_lean=phase_mean(pfms, "torso_lean"),
                arm_swing_left=phase_mean(pfms, "arm_swing_left"),
                arm_swing_right=phase_mean(pfms, "arm_swing_right"),
                cadence=0.0,
                timestamp=_t.time(),
            )
            pa = AnomalyResult(
                is_anomaly=sum(f.is_anomaly for f in pfms) / len(pfms) > 0.3,
                ae_error=0.0,
            )
            msgs = feedback_gen.generate(pf, pa, phase=phase)
            for m in msgs:
                key = f"{m.feature}:{m.level}:{m.message[:30]}"
                if key not in seen:
                    seen.add(key)
                    feedback_msgs.append({
                        "feature": m.feature,
                        "level": m.level,
                        "message": m.message,
                        "suggestion": m.suggestion,
                    })

        if not feedback_msgs:
            from feedback_generator import GOOD_MESSAGE as GM
            feedback_msgs = [{
                "feature": GM.feature, "level": GM.level,
                "message": GM.message, "suggestion": GM.suggestion,
            }]

    timeline = [asdict(f) for f in frame_metrics]

    return {
        "duration": round(total / fps, 2),
        "fps":      round(fps, 2),
        "metrics":  metrics,
        "feedback": feedback_msgs,
        "timeline": timeline,
    }
