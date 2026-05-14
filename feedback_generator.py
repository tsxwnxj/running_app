"""룰 기반 한국어 피드백 생성.

각 피처의 이탈 방향을 보행 단계별 패턴과 연결해 구체적인 교정 방법 제시.
MediaPipe 각도 기준:
  knee_angle  = ∠(Hip-Knee-Ankle),  180° = 완전 신전, 값 감소 = 굴곡 증가
  hip_angle   = ∠(Shoulder-Hip-Knee), trunk-thigh angle
  torso_lean  = 수직축 대비 상체 기울기, 양수 = 앞으로 기울어짐
  arm_swing   = ∠(Shoulder-Elbow-Wrist)
"""

from dataclasses import dataclass
from typing import Optional
from feature_extractor import RunningFeatures
from anomaly_detector import AnomalyResult, NORMAL_RANGES, _get_range
from gait_phase import GaitPhase, PHASE_LABELS_KO


@dataclass
class FeedbackMessage:
    feature: str
    level: str        # "good" | "warning" | "error"
    message: str
    suggestion: str


GOOD_MESSAGE = FeedbackMessage(
    feature="overall",
    level="good",
    message="자세가 좋습니다!",
    suggestion="현재 자세를 유지하세요.",
)


# ── 무릎 각도 ────────────────────────────────────────────────
def _knee_feedback(features: RunningFeatures, phase: Optional[GaitPhase] = None) -> list[FeedbackMessage]:
    msgs = []
    phase_ko = PHASE_LABELS_KO.get(phase, "") if phase else ""
    phase_tag = f"[{phase_ko}] " if phase_ko else ""

    pairs = [("왼쪽", "left", features.knee_angle_left), ("오른쪽", "right", features.knee_angle_right)]

    for side_ko, side_en, angle in pairs:
        if angle is None:
            continue
        lo, hi = _get_range("knee_angle", phase, side_en)
        deviation = max(0.0, lo - angle, angle - hi)
        if deviation < 5.0:
            continue
        level = "error" if deviation >= 10.0 else "warning"

        if angle > hi:
            msgs.append(FeedbackMessage(
                feature="knee_angle",
                level=level,
                message=f"{phase_tag}{side_ko} 무릎이 충분히 굽혀지지 않았습니다. ({angle:.0f}°, 정상 {lo:.0f}~{hi:.0f}°)",
                suggestion="착지 시 무릎을 더 굽혀 충격을 분산하세요. Overstride 여부를 확인하고 착지 발을 무게중심 아래에 위치시키세요.",
            ))
        elif angle < lo:
            msgs.append(FeedbackMessage(
                feature="knee_angle",
                level=level,
                message=f"{phase_tag}{side_ko} 무릎이 과도하게 굽혀져 있습니다. ({angle:.0f}°, 정상 {lo:.0f}~{hi:.0f}°)",
                suggestion="보폭 리듬을 점검하고 과도한 햄스트링 수축을 줄여보세요.",
            ))

    l_k, r_k = features.knee_angle_left, features.knee_angle_right
    if l_k is not None and r_k is not None and abs(l_k - r_k) > 15:
        msgs.append(FeedbackMessage(
            feature="knee_angle",
            level="warning",
            message=f"{phase_tag}무릎 각도 좌우 비대칭이 큽니다. (좌 {l_k:.0f}° / 우 {r_k:.0f}°, 차이 {abs(l_k-r_k):.0f}°)",
            suggestion="한쪽 발에 과부하가 걸릴 수 있습니다. 양발 착지 패턴이 균일한지 확인하세요.",
        ))
    return msgs


# ── 고관절 각도 ───────────────────────────────────────────────
def _hip_feedback(features: RunningFeatures, phase: Optional[GaitPhase] = None) -> list[FeedbackMessage]:
    msgs = []
    phase_ko = PHASE_LABELS_KO.get(phase, "") if phase else ""
    phase_tag = f"[{phase_ko}] " if phase_ko else ""

    lo, hi = _get_range("hip_angle", phase)
    pairs = [("왼쪽", features.hip_angle_left), ("오른쪽", features.hip_angle_right)]

    for side, angle in pairs:
        if angle is None:
            continue
        deviation = max(0.0, lo - angle, angle - hi)
        if deviation < 5.0:
            continue
        level = "error" if deviation >= 10.0 else "warning"

        if angle > hi:
            msgs.append(FeedbackMessage(
                feature="hip_angle",
                level=level,
                message=f"{phase_tag}{side} 고관절 신전이 부족합니다. ({angle:.0f}°, 정상 {lo:.0f}~{hi:.0f}°)",
                suggestion="Toe Off 시 고관절을 충분히 펴야 추진력을 얻을 수 있습니다. 힙 플렉서 스트레칭과 글루트 활성화 운동을 권장합니다.",
            ))
        elif angle < lo:
            msgs.append(FeedbackMessage(
                feature="hip_angle",
                level=level,
                message=f"{phase_tag}{side} 고관절이 과도하게 굽혀져 있습니다. ({angle:.0f}°, 정상 {lo:.0f}~{hi:.0f}°)",
                suggestion="상체가 앞으로 쏠리거나 보폭이 과도할 수 있습니다. 코어에 힘을 주고 상체를 세우세요.",
            ))

    l, r = features.hip_angle_left, features.hip_angle_right
    if l is not None and r is not None and abs(l - r) > 12:
        msgs.append(FeedbackMessage(
            feature="hip_angle",
            level="warning",
            message=f"{phase_tag}고관절 각도 좌우 비대칭이 있습니다. (좌 {l:.0f}° / 우 {r:.0f}°)",
            suggestion="골반 안정성 문제일 수 있습니다. 단다리 서기, 클램셸 운동으로 엉덩이 근력을 강화하세요.",
        ))
    return msgs


# ── 상체 기울기 ───────────────────────────────────────────────
def _torso_feedback(features: RunningFeatures, phase: Optional[GaitPhase] = None) -> list[FeedbackMessage]:
    msgs = []
    lean = features.torso_lean
    if lean is None:
        return msgs

    phase_ko = PHASE_LABELS_KO.get(phase, "") if phase else ""
    phase_tag = f"[{phase_ko}] " if phase_ko else ""
    lo, hi = _get_range("torso_lean", phase)

    deviation = max(0.0, lo - lean, lean - hi)
    if deviation >= 5.0:
        level = "error" if deviation >= 10.0 else "warning"
        if lean > hi:
            msgs.append(FeedbackMessage(
                feature="torso_lean",
                level=level,
                message=f"{phase_tag}상체가 앞으로 과도하게 기울어져 있습니다. ({lean:.1f}°, 정상 {lo:.0f}~{hi:.0f}°)",
                suggestion="코어 근육에 힘을 주고 시선을 전방 20~30m에 고정하세요. 피로 시 나타나기 쉬우니 주의하세요.",
            ))
        elif lean < lo:
            msgs.append(FeedbackMessage(
                feature="torso_lean",
                level=level,
                message=f"{phase_tag}상체 기울기가 부족합니다. ({lean:.1f}°, 정상 {lo:.0f}~{hi:.0f}°)",
                suggestion="러닝 시 5~10° 전방 기울기가 추진에 도움이 됩니다. 발목 위에 엉덩이가 위치하도록 자세를 조정하세요.",
            ))
    return msgs


# ── 팔 스윙 ──────────────────────────────────────────────────
def _arm_swing_feedback(features: RunningFeatures, phase: Optional[GaitPhase] = None) -> list[FeedbackMessage]:
    msgs = []
    phase_ko = PHASE_LABELS_KO.get(phase, "") if phase else ""
    phase_tag = f"[{phase_ko}] " if phase_ko else ""
    lo, hi = _get_range("arm_swing", phase)

    pairs = [("왼쪽", features.arm_swing_left), ("오른쪽", features.arm_swing_right)]
    for side, angle in pairs:
        if angle is None:
            continue
        deviation = max(0.0, lo - angle, angle - hi)
        if deviation < 5.0:
            continue
        level = "error" if deviation >= 10.0 else "warning"

        if angle < lo:
            msgs.append(FeedbackMessage(
                feature="arm_swing",
                level=level,
                message=f"{phase_tag}{side} 팔꿈치가 과도하게 굽혀져 있습니다. ({angle:.0f}°, 정상 {lo:.0f}~{hi:.0f}°)",
                suggestion="팔꿈치 각도를 50~100°로 유지하세요. 너무 많이 굽히면 상체 긴장을 유발합니다.",
            ))
        elif angle > hi:
            msgs.append(FeedbackMessage(
                feature="arm_swing",
                level=level,
                message=f"{phase_tag}{side} 팔이 너무 펴져 있습니다. ({angle:.0f}°, 정상 {lo:.0f}~{hi:.0f}°)",
                suggestion="팔꿈치를 약 90° 구부린 상태로 앞뒤로 흔드세요. 팔이 펴지면 에너지 낭비가 생깁니다.",
            ))

    l, r = features.arm_swing_left, features.arm_swing_right
    if l is not None and r is not None and abs(l - r) > 20:
        msgs.append(FeedbackMessage(
            feature="arm_swing",
            level="warning",
            message=f"{phase_tag}팔 스윙 좌우 비대칭이 있습니다. (좌 {l:.0f}° / 우 {r:.0f}°)",
            suggestion="Cross swing 가능성이 있습니다. 팔을 몸 정중선을 넘지 않게 앞뒤로만 흔드세요.",
        ))
    return msgs


# ── 케이던스 ──────────────────────────────────────────────────
def _cadence_feedback(features: RunningFeatures) -> list[FeedbackMessage]:
    if features.cadence <= 0:
        return []
    cad = features.cadence
    lo, hi = NORMAL_RANGES["cadence"]

    if cad < lo:
        return [FeedbackMessage(
            feature="cadence",
            level="warning",
            message=f"케이던스가 낮습니다. ({cad:.0f} spm)",
            suggestion="보폭을 줄이고 발을 더 빠르게 내딛어 170~180 spm을 목표로 하세요. 낮은 케이던스는 Overstride와 연관됩니다.",
        )]
    elif cad > hi:
        return [FeedbackMessage(
            feature="cadence",
            level="warning",
            message=f"케이던스가 매우 높습니다. ({cad:.0f} spm)",
            suggestion="케이던스가 과도하게 높으면 보폭이 너무 짧아질 수 있습니다. 속도에 맞는 자연스러운 리듬을 찾아보세요.",
        )]
    return []


class FeedbackGenerator:
    def generate(
        self,
        features: RunningFeatures,
        anomaly: AnomalyResult,
        phase: Optional[GaitPhase] = None,
    ) -> list[FeedbackMessage]:
        if not anomaly.is_anomaly:
            return [GOOD_MESSAGE]

        msgs: list[FeedbackMessage] = []
        msgs.extend(_knee_feedback(features, phase))
        msgs.extend(_hip_feedback(features, phase))
        msgs.extend(_torso_feedback(features, phase))
        msgs.extend(_arm_swing_feedback(features, phase))

        severity = {"error": 0, "warning": 1, "good": 2}
        msgs.sort(key=lambda m: severity[m.level])

        return msgs if msgs else [GOOD_MESSAGE]
