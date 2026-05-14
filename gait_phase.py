"""보행 단계 감지.

좌우 무릎 각도를 비교해 어느 쪽 무릎이 최전방인지 프레임별로 분류.
  - LEFT_KNEE_FORWARD:  왼쪽 무릎 최전방 (= 오른쪽 무릎 최후방, 오른팔 최전방)
  - RIGHT_KNEE_FORWARD: 오른쪽 무릎 최전방 (= 왼쪽 무릎 최후방, 왼팔 최전방)
"""

from enum import Enum
from typing import Optional


class GaitPhase(Enum):
    LEFT_KNEE_FORWARD  = "LKF"
    RIGHT_KNEE_FORWARD = "RKF"
    UNKNOWN            = "UNK"


PHASE_LABELS_KO = {
    GaitPhase.LEFT_KNEE_FORWARD:  "왼무릎 최전방",
    GaitPhase.RIGHT_KNEE_FORWARD: "오른무릎 최전방",
    GaitPhase.UNKNOWN:            "",
}


class GaitPhaseDetector:
    """
    무릎 각도(Hip-Knee-Ankle) 비교로 단계 분류.
    더 신전된(각도 큰) 무릎이 현재 앞으로 나온 무릎.
    """

    def __init__(self, fps: float = 30.0):
        self.fps = fps

    def detect(self, smoothed_landmarks: list[Optional[dict]]) -> list[GaitPhase]:
        from feature_extractor import _angle_3points

        phases: list[GaitPhase] = []
        for lm in smoothed_landmarks:
            if lm is None:
                phases.append(GaitPhase.UNKNOWN)
                continue

            l_ang: Optional[float] = None
            r_ang: Optional[float] = None

            if all(k in lm for k in ["left_hip", "left_knee", "left_ankle"]):
                l_ang = _angle_3points(lm["left_hip"], lm["left_knee"], lm["left_ankle"])
            if all(k in lm for k in ["right_hip", "right_knee", "right_ankle"]):
                r_ang = _angle_3points(lm["right_hip"], lm["right_knee"], lm["right_ankle"])

            if l_ang is not None and r_ang is not None:
                phases.append(
                    GaitPhase.LEFT_KNEE_FORWARD if l_ang >= r_ang
                    else GaitPhase.RIGHT_KNEE_FORWARD
                )
            elif l_ang is not None:
                # 왼쪽만 보임: 각도 크면 왼쪽이 앞, 작으면 뒤(= 오른쪽이 앞)
                phases.append(
                    GaitPhase.LEFT_KNEE_FORWARD if l_ang > 120
                    else GaitPhase.RIGHT_KNEE_FORWARD
                )
            elif r_ang is not None:
                phases.append(
                    GaitPhase.RIGHT_KNEE_FORWARD if r_ang > 120
                    else GaitPhase.LEFT_KNEE_FORWARD
                )
            else:
                phases.append(GaitPhase.UNKNOWN)

        return phases
