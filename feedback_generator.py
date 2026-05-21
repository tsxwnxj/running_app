"""룰 기반 한국어 피드백 생성.

실측 정상 범위 (학습 데이터 5th~95th pct):
  knee_angle (전방 무릎) : 125~172°   knee_angle (후방 무릎) : 55~165°
  hip_angle              : 130~176°   torso_lean             : 5~15°
  arm_swing              : 70~130°

이탈 기준:
  5° 미만 이탈   → 정상 (피드백 없음)
  5~25° 이탈    → 주의 (warning)
  25° 초과 이탈  → 위험 (error)
"""

from dataclasses import dataclass
from typing import Optional
from feature_extractor import RunningFeatures
from anomaly_detector import AnomalyResult, NORMAL_RANGES, _get_range
from gait_phase import GaitPhase, PHASE_LABELS_KO


@dataclass
class FeedbackMessage:
    feature: str
    level: str                    # "good" | "warning" | "error"
    message: str
    suggestion: str
    timestamp: Optional[float] = None  # 이탈이 가장 심한 프레임 시각(초)


GOOD_MESSAGE = FeedbackMessage(
    feature="overall",
    level="good",
    message="자세가 좋습니다!",
    suggestion="현재 자세를 유지하세요.",
)


# ── 무릎 각도 ────────────────────────────────────────────────────
def _knee_feedback(features: RunningFeatures, phase: Optional[GaitPhase] = None) -> list[FeedbackMessage]:
    msgs = []
    phase_ko  = PHASE_LABELS_KO.get(phase, "") if phase else ""
    phase_tag = f"[{phase_ko}] " if phase_ko else ""

    pairs = [("왼쪽", "left", features.knee_angle_left), ("오른쪽", "right", features.knee_angle_right)]

    for side_ko, side_en, angle in pairs:
        if angle is None:
            continue
        lo, hi = _get_range("knee_angle", phase, side_en)
        deviation = max(0.0, lo - angle, angle - hi)
        if deviation < 5.0:
            continue
        level = "error" if deviation >= 25.0 else "warning"

        # 전방 무릎(125~172°) vs 후방 무릎(55~165°) 구분해서 메시지
        is_forward = (
            (phase == GaitPhase.LEFT_KNEE_FORWARD  and side_en == "left") or
            (phase == GaitPhase.RIGHT_KNEE_FORWARD and side_en == "right")
        )

        if angle > hi:
            if is_forward:
                msgs.append(FeedbackMessage(
                    feature="knee_angle",
                    level=level,
                    message=f"{phase_tag}{side_ko} 무릎이 과도하게 펴진 채 착지하고 있습니다. ({angle:.0f}°, 정상 {lo:.0f}~{hi:.0f}°)",
                    suggestion="착지 순간 무릎을 살짝 굽혀 충격을 흡수하세요. 무릎이 너무 펴지면 관절에 충격이 직접 전달됩니다.",
                ))
            else:
                msgs.append(FeedbackMessage(
                    feature="knee_angle",
                    level=level,
                    message=f"{phase_tag}{side_ko} 무릎이 스윙 시 충분히 접히지 않습니다. ({angle:.0f}°, 정상 {lo:.0f}~{hi:.0f}°)",
                    suggestion="발을 뒤로 차올릴 때 햄스트링을 적극적으로 수축시켜 무릎을 더 접어주세요. 스트라이드 효율이 낮아질 수 있습니다.",
                ))
        elif angle < lo:
            if is_forward:
                msgs.append(FeedbackMessage(
                    feature="knee_angle",
                    level=level,
                    message=f"{phase_tag}{side_ko} 무릎이 착지 시 과도하게 굽혀져 있습니다. ({angle:.0f}°, 정상 {lo:.0f}~{hi:.0f}°)",
                    suggestion="착지 발이 무게중심보다 너무 앞으로 나가고 있을 수 있습니다. 보폭을 줄이고 발을 엉덩이 아래쪽에 착지시키세요.",
                ))
            else:
                msgs.append(FeedbackMessage(
                    feature="knee_angle",
                    level=level,
                    message=f"{phase_tag}{side_ko} 무릎이 스윙 구간에서 지나치게 접혀 있습니다. ({angle:.0f}°, 정상 {lo:.0f}~{hi:.0f}°)",
                    suggestion="대퇴사두근 긴장이 과도하거나 보폭이 짧을 수 있습니다. 자연스러운 진자 운동으로 다리를 앞으로 보내세요.",
                ))

    return msgs


# ── 고관절 각도 ───────────────────────────────────────────────────
def _hip_feedback(features: RunningFeatures, phase: Optional[GaitPhase] = None) -> list[FeedbackMessage]:
    msgs = []
    phase_ko  = PHASE_LABELS_KO.get(phase, "") if phase else ""
    phase_tag = f"[{phase_ko}] " if phase_ko else ""

    lo, hi = _get_range("hip_angle", phase)
    pairs = [("왼쪽", features.hip_angle_left), ("오른쪽", features.hip_angle_right)]

    for side, angle in pairs:
        if angle is None:
            continue
        deviation = max(0.0, lo - angle, angle - hi)
        if deviation < 5.0:
            continue
        level = "error" if deviation >= 25.0 else "warning"

        if angle > hi:
            msgs.append(FeedbackMessage(
                feature="hip_angle",
                level=level,
                message=f"{phase_tag}{side} 고관절이 충분히 펴지지 않아 추진력이 부족합니다. ({angle:.0f}°, 정상 {lo:.0f}~{hi:.0f}°)",
                suggestion="발이 지면을 떠날 때 엉덩이를 뒤로 충분히 밀어주세요. 힙 플렉서 스트레칭과 글루트 브릿지 운동으로 고관절 가동 범위를 늘리세요.",
            ))
        elif angle < lo:
            msgs.append(FeedbackMessage(
                feature="hip_angle",
                level=level,
                message=f"{phase_tag}{side} 고관절이 과도하게 굽혀져 있습니다. ({angle:.0f}°, 정상 {lo:.0f}~{hi:.0f}°)",
                suggestion="상체가 앞으로 쏠리거나 보폭이 과도하게 클 수 있습니다. 코어에 힘을 주고 골반을 중립 위치로 유지하세요.",
            ))

    return msgs


# ── 상체 기울기 ───────────────────────────────────────────────────
def _torso_feedback(features: RunningFeatures, phase: Optional[GaitPhase] = None) -> list[FeedbackMessage]:
    msgs = []
    lean = features.torso_lean
    if lean is None:
        return msgs

    phase_ko  = PHASE_LABELS_KO.get(phase, "") if phase else ""
    phase_tag = f"[{phase_ko}] " if phase_ko else ""
    lo, hi    = _get_range("torso_lean", phase)

    deviation = max(0.0, lo - lean, lean - hi)
    if deviation < 5.0:
        return msgs
    level = "error" if deviation >= 25.0 else "warning"

    if lean > hi:
        msgs.append(FeedbackMessage(
            feature="torso_lean",
            level=level,
            message=f"{phase_tag}상체가 앞으로 과도하게 기울어져 있습니다. ({lean:.1f}°, 정상 {lo:.0f}~{hi:.0f}°)",
            suggestion="코어 근육에 힘을 주고 시선을 전방 20~30m에 고정하세요. 과도한 앞 기울기는 허리와 햄스트링 부상 위험을 높입니다.",
        ))
    elif lean < lo:
        msgs.append(FeedbackMessage(
            feature="torso_lean",
            level=level,
            message=f"{phase_tag}상체가 너무 세워져 있거나 뒤로 젖혀지고 있습니다. ({lean:.1f}°, 정상 {lo:.0f}~{hi:.0f}°)",
            suggestion="5~15° 정도 자연스럽게 앞으로 기울어야 추진 효율이 높아집니다. 발목 위에 엉덩이, 엉덩이 위에 어깨가 오도록 정렬하세요.",
        ))
    return msgs


# ── 팔 스윙 ──────────────────────────────────────────────────────
def _arm_swing_feedback(features: RunningFeatures, phase: Optional[GaitPhase] = None) -> list[FeedbackMessage]:
    msgs = []
    phase_ko  = PHASE_LABELS_KO.get(phase, "") if phase else ""
    phase_tag = f"[{phase_ko}] " if phase_ko else ""
    lo, hi    = _get_range("arm_swing", phase)

    pairs = [("왼쪽", features.arm_swing_left), ("오른쪽", features.arm_swing_right)]
    for side, angle in pairs:
        if angle is None:
            continue
        deviation = max(0.0, lo - angle, angle - hi)
        if deviation < 5.0:
            continue
        level = "error" if deviation >= 25.0 else "warning"

        if angle < lo:
            msgs.append(FeedbackMessage(
                feature="arm_swing",
                level=level,
                message=f"{phase_tag}{side} 팔꿈치가 과도하게 굽혀져 있습니다. ({angle:.0f}°, 정상 {lo:.0f}~{hi:.0f}°)",
                suggestion="팔꿈치를 70~130° 범위로 유지하세요. 지나치게 굽히면 어깨와 목에 불필요한 긴장이 생깁니다.",
            ))
        elif angle > hi:
            msgs.append(FeedbackMessage(
                feature="arm_swing",
                level=level,
                message=f"{phase_tag}{side} 팔이 너무 펴진 채 스윙되고 있습니다. ({angle:.0f}°, 정상 {lo:.0f}~{hi:.0f}°)",
                suggestion="팔꿈치를 약 90° 기준으로 굽히고 앞뒤로만 흔드세요. 팔이 펴지면 회전 모멘트가 커져 에너지 손실이 발생합니다.",
            ))

    l, r = features.arm_swing_left, features.arm_swing_right
    if l is not None and r is not None and abs(l - r) > 20:
        msgs.append(FeedbackMessage(
            feature="arm_swing",
            level="warning",
            message=f"{phase_tag}팔 스윙 좌우 비대칭이 있습니다. (좌 {l:.0f}° / 우 {r:.0f}°, 차이 {abs(l-r):.0f}°)",
            suggestion="한쪽 팔이 몸 안쪽으로 과하게 들어오는 Cross swing일 수 있습니다. 팔을 정중선을 넘지 않게 앞뒤로만 흔드세요.",
        ))
    return msgs


class FeedbackGenerator:
    def generate(
        self,
        features: RunningFeatures,
        anomaly: AnomalyResult,
        phase: Optional[GaitPhase] = None,
    ) -> list[FeedbackMessage]:
        # 피드백은 규칙 기반 위반 여부로 판단 (AE와 독립적)
        msgs: list[FeedbackMessage] = []
        msgs.extend(_knee_feedback(features, phase))
        msgs.extend(_hip_feedback(features, phase))
        msgs.extend(_torso_feedback(features, phase))
        msgs.extend(_arm_swing_feedback(features, phase))

        severity = {"error": 0, "warning": 1, "good": 2}
        msgs.sort(key=lambda m: severity[m.level])

        return msgs if msgs else [GOOD_MESSAGE]
