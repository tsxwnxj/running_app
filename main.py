"""2D 포즈 추정을 활용한 실시간 러닝 자세 피드백 시스템.

실행: python main.py [--source 0] [--width 1280] [--height 720]
"""

import argparse
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# 한국어 폰트 렌더링 (PIL 사용)
try:
    from PIL import ImageFont, ImageDraw, Image as PILImage
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

sys.path.insert(0, str(Path(__file__).parent))
from pose_estimator import PoseEstimator
from feature_extractor import FeatureExtractor, RunningFeatures
from anomaly_detector import AnomalyDetector, AnomalyResult
from feedback_generator import FeedbackGenerator, FeedbackMessage

# ── 색상 팔레트 (BGR) ─────────────────────────────────────────
COLOR_BG        = (20, 20, 20)
COLOR_PANEL     = (35, 35, 35)
COLOR_GOOD      = (80, 200, 80)
COLOR_WARNING   = (40, 180, 255)
COLOR_ERROR     = (60, 60, 230)
COLOR_TEXT      = (240, 240, 240)
COLOR_SUBTEXT   = (160, 160, 160)
COLOR_ACCENT    = (255, 160, 40)
COLOR_BORDER    = (70, 70, 70)

PANEL_W = 400           # 오른쪽 정보 패널 너비
FPS_SMOOTH = 10         # FPS 평균 윈도우


def find_korean_font() -> Optional[str]:
    candidates = [
        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/Library/Fonts/NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    return None


class KoreanRenderer:
    """PIL을 이용한 한국어 텍스트 렌더링."""

    def __init__(self, font_path: Optional[str], size: int = 18):
        self._font_path = font_path
        self._cache: dict[int, object] = {}
        self._size = size

    def _get_font(self, size: int):
        if size not in self._cache:
            if _HAS_PIL and self._font_path:
                self._cache[size] = ImageFont.truetype(self._font_path, size)
            else:
                self._cache[size] = None
        return self._cache[size]

    def put_text(
        self,
        img: np.ndarray,
        text: str,
        pos: tuple[int, int],
        size: int = 18,
        color: tuple[int, int, int] = COLOR_TEXT,
        bold: bool = False,
    ) -> np.ndarray:
        if not _HAS_PIL or not self._font_path:
            cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, size / 30, color[::-1], 1, cv2.LINE_AA)
            return img

        pil = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        font = self._get_font(size)
        draw.text(pos, text, font=font, fill=(color[2], color[1], color[0]))
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def level_color(level: str) -> tuple[int, int, int]:
    return {"good": COLOR_GOOD, "warning": COLOR_WARNING, "error": COLOR_ERROR}.get(level, COLOR_TEXT)


def draw_rounded_rect(img, x, y, w, h, r, color, thickness=-1):
    if thickness == -1:
        cv2.rectangle(img, (x + r, y), (x + w - r, y + h), color, -1)
        cv2.rectangle(img, (x, y + r), (x + w, y + h - r), color, -1)
        for cx, cy in [(x + r, y + r), (x + w - r, y + r), (x + r, y + h - r), (x + w - r, y + h - r)]:
            cv2.circle(img, (cx, cy), r, color, -1)
    else:
        cv2.rectangle(img, (x + r, y), (x + w - r, y + h), color, thickness)
        cv2.rectangle(img, (x, y + r), (x + w, y + h - r), color, thickness)
        for cx, cy in [(x + r, y + r), (x + w - r, y + r), (x + r, y + h - r), (x + w - r, y + h - r)]:
            cv2.ellipse(img, (cx, cy), (r, r), 0, 0, 360, color, thickness)


def draw_gauge(img, x, y, w, h, score: float, label: str, renderer: KoreanRenderer):
    """가로 게이지 바 (score: 0~1)."""
    bg_color = (60, 60, 60)
    cv2.rectangle(img, (x, y), (x + w, y + h), bg_color, -1)

    fill_w = int(w * max(0.0, min(1.0, score)))
    if score >= 0.85:
        bar_color = COLOR_GOOD
    elif score >= 0.55:
        bar_color = COLOR_WARNING
    else:
        bar_color = COLOR_ERROR
    if fill_w > 0:
        cv2.rectangle(img, (x, y), (x + fill_w, y + h), bar_color, -1)

    cv2.rectangle(img, (x, y), (x + w, y + h), COLOR_BORDER, 1)
    img = renderer.put_text(img, label, (x, y - 20), size=15, color=COLOR_SUBTEXT)
    pct_text = f"{int(score * 100)}%"
    img = renderer.put_text(img, pct_text, (x + w + 6, y - 2), size=14, color=bar_color)
    return img


def draw_panel(
    canvas: np.ndarray,
    features: Optional[RunningFeatures],
    anomaly: Optional[AnomalyResult],
    feedbacks: list[FeedbackMessage],
    scores: dict[str, float],
    fps: float,
    renderer: KoreanRenderer,
    panel_x: int,
) -> np.ndarray:
    h, w = canvas.shape[:2]
    cv2.rectangle(canvas, (panel_x, 0), (w, h), COLOR_PANEL, -1)
    cv2.line(canvas, (panel_x, 0), (panel_x, h), COLOR_BORDER, 2)

    px = panel_x + 16
    py = 18

    # ── 타이틀 ────────────────────────────────────────────────
    canvas = renderer.put_text(canvas, "러닝 자세 분석", (px, py), size=22, color=COLOR_ACCENT, bold=True)
    py += 30
    canvas = renderer.put_text(canvas, f"FPS: {fps:.1f}", (px, py), size=14, color=COLOR_SUBTEXT)
    py += 28
    cv2.line(canvas, (panel_x + 8, py), (w - 8, py), COLOR_BORDER, 1)
    py += 14

    if features is None:
        canvas = renderer.put_text(canvas, "사람을 감지하는 중...", (px, py), size=17, color=COLOR_WARNING)
        return canvas

    # ── 종합 상태 ──────────────────────────────────────────────
    if anomaly and anomaly.is_anomaly:
        status_text, status_color = "자세 이상 감지", COLOR_ERROR
    else:
        status_text, status_color = "정상 자세", COLOR_GOOD

    draw_rounded_rect(canvas, px - 6, py - 2, PANEL_W - 28, 32, 6, status_color)
    canvas = renderer.put_text(canvas, status_text, (px + 4, py + 4), size=18, color=(10, 10, 10), bold=True)
    py += 44

    # ── 피처 수치 ──────────────────────────────────────────────
    canvas = renderer.put_text(canvas, "[ 관절 각도 ]", (px, py), size=15, color=COLOR_SUBTEXT)
    py += 22

    if features:
        rows = [
            ("무릎 각도 (좌)", features.knee_angle_left, "°"),
            ("무릎 각도 (우)", features.knee_angle_right, "°"),
            ("고관절 (좌)",    features.hip_angle_left, "°"),
            ("고관절 (우)",    features.hip_angle_right, "°"),
            ("상체 기울기",    features.torso_lean, "°"),
            ("팔 스윙 (좌)",   features.arm_swing_left, "°"),
            ("팔 스윙 (우)",   features.arm_swing_right, "°"),
            ("케이던스",       features.cadence, " spm"),
        ]
        for label, val, unit in rows:
            val_str = f"{val:.1f}{unit}"
            canvas = renderer.put_text(canvas, label, (px, py), size=14, color=COLOR_SUBTEXT)
            canvas = renderer.put_text(canvas, val_str, (px + 160, py), size=14, color=COLOR_TEXT)
            py += 20

    py += 8
    cv2.line(canvas, (panel_x + 8, py), (w - 8, py), COLOR_BORDER, 1)
    py += 14

    # ── 피처별 게이지 ─────────────────────────────────────────
    canvas = renderer.put_text(canvas, "[ 자세 점수 ]", (px, py), size=15, color=COLOR_SUBTEXT)
    py += 28

    gauge_w = PANEL_W - 80
    gauge_h = 12
    gauge_labels = {
        "knee_angle": "무릎 각도",
        "hip_angle":  "고관절",
        "torso_lean": "상체 기울기",
        "arm_swing":  "팔 스윙",
        "cadence":    "케이던스",
    }
    for key, label in gauge_labels.items():
        score = scores.get(key, 1.0)
        canvas = draw_gauge(canvas, px, py, gauge_w, gauge_h, score, label, renderer)
        py += 38

    cv2.line(canvas, (panel_x + 8, py), (w - 8, py), COLOR_BORDER, 1)
    py += 14

    # ── 피드백 메시지 ─────────────────────────────────────────
    canvas = renderer.put_text(canvas, "[ 피드백 ]", (px, py), size=15, color=COLOR_SUBTEXT)
    py += 22

    max_msgs = 4
    for msg in feedbacks[:max_msgs]:
        col = level_color(msg.level)
        indicator = {"good": "●", "warning": "▲", "error": "■"}.get(msg.level, "●")
        canvas = renderer.put_text(canvas, indicator + " " + msg.message, (px, py), size=14, color=col)
        py += 20
        canvas = renderer.put_text(canvas, "  → " + msg.suggestion, (px, py), size=13, color=COLOR_SUBTEXT)
        py += 22

    if anomaly:
        py += 6
        canvas = renderer.put_text(
            canvas,
            f"AE 오차: {anomaly.ae_error:.3f}",
            (px, py), size=13, color=COLOR_SUBTEXT,
        )

    return canvas


def run(source: int | str, width: int, height: int):
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        print(f"[Error] 카메라/영상을 열 수 없습니다: {source}")
        sys.exit(1)

    estimator = PoseEstimator()
    extractor = FeatureExtractor()
    detector = AnomalyDetector()
    feedback_gen = FeedbackGenerator()

    font_path = find_korean_font()
    renderer = KoreanRenderer(font_path)

    if not _HAS_PIL:
        print("[Warning] Pillow 미설치 — 한국어 폰트 비활성화 (pip install Pillow)")
    elif not font_path:
        print("[Warning] 한국어 폰트를 찾을 수 없습니다.")

    fps_times: deque[float] = deque(maxlen=FPS_SMOOTH)
    last_features: Optional[RunningFeatures] = None
    last_anomaly: Optional[AnomalyResult] = None
    last_feedbacks: list[FeedbackMessage] = []
    last_scores: dict[str, float] = {}

    total_w = width + PANEL_W
    cv2.namedWindow("러닝 자세 분석", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("러닝 자세 분석", total_w, height)

    print("[Info] 실행 중... 종료: q 또는 ESC")

    while True:
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            print("[Info] 영상이 종료되었습니다.")
            break

        frame = cv2.resize(frame, (width, height))
        pose_result = estimator.process(frame)

        if pose_result:
            annotated = estimator.draw_skeleton(frame, pose_result)
            features = extractor.extract(pose_result)
            if features:
                anomaly = detector.detect(features)
                feedbacks = feedback_gen.generate(features, anomaly)
                scores = detector.get_per_feature_score(features)
                last_features = features
                last_anomaly = anomaly
                last_feedbacks = feedbacks
                last_scores = scores
        else:
            annotated = frame

        # 이상 감지 시 경계선 표시
        if last_anomaly and last_anomaly.is_anomaly:
            cv2.rectangle(annotated, (0, 0), (width - 1, height - 1), COLOR_ERROR, 4)

        # 캔버스 구성 (카메라 | 패널)
        canvas = np.zeros((height, total_w, 3), dtype=np.uint8)
        canvas[:, :width] = annotated
        canvas = draw_panel(
            canvas, last_features, last_anomaly,
            last_feedbacks, last_scores,
            sum(fps_times) / max(len(fps_times), 1),
            renderer, width,
        )

        t1 = time.perf_counter()
        fps_times.append(1.0 / max(t1 - t0, 1e-6))

        cv2.imshow("러닝 자세 분석", canvas)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cap.release()
    estimator.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="실시간 러닝 자세 피드백 시스템")
    parser.add_argument("--source", default="0", help="카메라 인덱스 또는 영상 파일 경로 (기본: 0)")
    parser.add_argument("--width",  type=int, default=854,  help="카메라 가로 해상도")
    parser.add_argument("--height", type=int, default=480,  help="카메라 세로 해상도")
    args = parser.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source
    run(src, args.width, args.height)
