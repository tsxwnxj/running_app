"""MediaPipe Tasks API 기반 포즈 추정 모듈.

IMAGE 모드 사용: 프레임별 독립 추정으로 내부 lag 없음.
스무딩은 video_processor.py에서 Gaussian 필터로 처리.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

MODEL_PATH = str(Path(__file__).parent / "pose_landmarker_full.task")


@dataclass
class Landmark:
    x: float
    y: float
    z: float
    visibility: float


@dataclass
class PoseResult:
    landmarks: dict[str, Landmark]        # 이미지 좌표 (가시성 판단 + 스켈레톤 드로잉용)
    world_landmarks: dict[str, Landmark]  # 3D 월드 좌표 (각도 계산용, 힙 중심 미터 단위)
    raw_result: object
    frame: np.ndarray


LANDMARK_NAMES = {
    "nose":             0,
    "left_shoulder":   11,
    "right_shoulder":  12,
    "left_elbow":      13,
    "right_elbow":     14,
    "left_wrist":      15,
    "right_wrist":     16,
    "left_hip":        23,
    "right_hip":       24,
    "left_knee":       25,
    "right_knee":      26,
    "left_ankle":      27,
    "right_ankle":     28,
    "left_heel":       29,
    "right_heel":      30,
    "left_foot_index": 31,
    "right_foot_index":32,
}

SKELETON_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31),
    (24, 26), (26, 28), (28, 30), (28, 32),
]


class PoseEstimator:
    def __init__(
        self,
        model_path: str = MODEL_PATH,
        min_detection_confidence: float = 0.45,
        min_tracking_confidence:  float = 0.35,
    ):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"모델 파일 없음: {model_path}")

        options = PoseLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
            running_mode=VisionTaskRunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=False,
        )
        self._landmarker = PoseLandmarker.create_from_options(options)

    def process(self, frame: np.ndarray, timestamp_ms: int = 0) -> Optional[PoseResult]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return None

        h, w = frame.shape[:2]
        norm_lm  = result.pose_landmarks[0]
        world_lm = result.pose_world_landmarks[0] if result.pose_world_landmarks else norm_lm

        landmarks: dict[str, Landmark] = {}
        world_landmarks: dict[str, Landmark] = {}
        for name, idx in LANDMARK_NAMES.items():
            lm = norm_lm[idx]
            landmarks[name] = Landmark(
                x=lm.x * w,
                y=lm.y * h,
                z=lm.z,
                visibility=lm.visibility if hasattr(lm, "visibility") else 1.0,
            )
            wlm = world_lm[idx]
            world_landmarks[name] = Landmark(
                x=wlm.x,
                y=wlm.y,
                z=wlm.z,
                visibility=lm.visibility if hasattr(lm, "visibility") else 1.0,
            )

        return PoseResult(landmarks=landmarks, world_landmarks=world_landmarks, raw_result=result, frame=frame)

    def draw_skeleton(self, frame: np.ndarray, pose_result: PoseResult) -> np.ndarray:
        annotated = frame.copy()
        lm = pose_result.landmarks
        idx_to_lm = {v: lm[k] for k, v in LANDMARK_NAMES.items()}

        for i, j in SKELETON_CONNECTIONS:
            if i in idx_to_lm and j in idx_to_lm:
                a, b = idx_to_lm[i], idx_to_lm[j]
                cv2.line(annotated, (int(a.x), int(a.y)), (int(b.x), int(b.y)),
                         (0, 200, 255), 2, cv2.LINE_AA)

        for pt in lm.values():
            cv2.circle(annotated, (int(pt.x), int(pt.y)), 4, (255, 255, 255), -1)
            cv2.circle(annotated, (int(pt.x), int(pt.y)), 4, (0, 150, 255), 1)

        return annotated

    def close(self):
        self._landmarker.close()
