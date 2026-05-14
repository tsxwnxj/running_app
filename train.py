"""좋은 러너 영상 데이터로 오토인코더 학습.

사용법:
  1. training_data/ 폴더에 좋은 자세의 러닝 영상(mp4 등)을 넣는다.
  2. python train.py
  3. 완료 후 model/running_ae.pt 가 교체됨.
"""

import sys
import os
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

TRAINING_DIR = Path(__file__).parent / "training_data"
MODEL_PATH   = Path(__file__).parent / "model" / "running_ae.pt"
VIDEO_EXTS   = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def extract_features_from_video(video_path: str) -> np.ndarray:
    """영상 1개에서 유효 피처 벡터 시퀀스 추출."""
    import cv2
    from pose_estimator import PoseEstimator
    from feature_extractor import FeatureExtractor

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"영상을 열 수 없습니다: {video_path}")

    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    estimator = PoseEstimator()
    extractor = FeatureExtractor()
    vectors   = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ts = frame_idx / fps
        pose = estimator.process(frame, int(ts * 1000))
        if pose:
            feat = extractor.extract(pose, video_ts=ts)
            if feat and feat.has_any():
                vectors.append(extractor.to_vector(feat))
        frame_idx += 1
        if frame_idx % 30 == 0:
            pct = int(frame_idx / max(total, 1) * 100)
            print(f"  {pct}% ({frame_idx}/{total} 프레임)", end="\r")

    cap.release()
    estimator.close()
    print()
    return np.array(vectors, dtype=np.float32) if vectors else np.empty((0, 8), dtype=np.float32)


def main():
    videos = [p for p in TRAINING_DIR.iterdir() if p.suffix.lower() in VIDEO_EXTS]
    if not videos:
        print(f"[오류] {TRAINING_DIR}/ 에 영상 파일이 없습니다.")
        print("  mp4, mov, avi, mkv 파일을 넣고 다시 실행하세요.")
        sys.exit(1)

    print(f"영상 {len(videos)}개 발견:\n" + "\n".join(f"  {v.name}" for v in videos))

    all_vectors = []
    for i, vp in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}] {vp.name} 분석 중...")
        try:
            vecs = extract_features_from_video(str(vp))
            print(f"  → 유효 프레임 {len(vecs)}개")
            if len(vecs) > 0:
                all_vectors.append(vecs)
        except Exception as e:
            print(f"  [경고] 건너뜀: {e}")

    if not all_vectors:
        print("\n[오류] 유효한 피처 데이터가 없습니다.")
        sys.exit(1)

    data = np.concatenate(all_vectors, axis=0)
    print(f"\n총 {len(data)}개 프레임 수집 완료.")

    if len(data) < 200:
        print("[경고] 데이터가 너무 적습니다 (200프레임 미만). 영상을 더 추가하세요.")

    # 학습
    print("\n오토인코더 학습 중...")
    from model.autoencoder import (
        RunningAutoencoder, FeatureNormalizer,
        train_autoencoder, save_model,
    )

    normalizer = FeatureNormalizer()
    model = RunningAutoencoder()
    final_loss = train_autoencoder(model, normalizer, data, epochs=300, lr=1e-3)
    print(f"학습 완료 — 최종 loss: {final_loss:.6f}")

    # 임계값 자동 계산: 학습 데이터 재구성 오차의 95 퍼센타일
    import torch
    normalized = torch.from_numpy(normalizer.transform(data))
    with torch.no_grad():
        errors = model.reconstruction_error(normalized).numpy()
    threshold = float(np.percentile(errors, 95))
    print(f"이상 탐지 임계값 (95th pct): {threshold:.6f}")

    # 저장
    MODEL_PATH.parent.mkdir(exist_ok=True)
    save_model(model, normalizer, str(MODEL_PATH))

    # 임계값도 별도 저장 (anomaly_detector.py에서 로드)
    threshold_path = MODEL_PATH.parent / "threshold.txt"
    threshold_path.write_text(str(threshold))

    print(f"\n모델 저장 완료: {MODEL_PATH}")
    print("이제 서버를 재시작하면 새 기준이 적용됩니다.")


if __name__ == "__main__":
    main()
