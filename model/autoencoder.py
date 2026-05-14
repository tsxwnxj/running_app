"""오토인코더 기반 이상 탐지 모델 (PyTorch).

정상 러닝 데이터로 학습 후 재구성 오차(MSE)로 이상 판단.
Running Injury Clinic Kinematic Dataset 등 실제 데이터로 학습하면 정확도 향상 가능.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

INPUT_DIM = 8  # feature_extractor.py의 to_vector() 출력 차원


class RunningAutoencoder(nn.Module):
    def __init__(self, input_dim: int = INPUT_DIM, latent_dim: int = 4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            recon = self(x)
            return torch.mean((recon - x) ** 2, dim=-1)


def generate_normal_running_data(n: int = 5000) -> np.ndarray:
    """정상 러닝 데이터 시뮬레이션 (실제 데이터셋 없을 때 사용).

    각 피처의 정상 범위는 운동역학 문헌 기반:
    [0] knee_angle_left   : 145–175°
    [1] knee_angle_right  : 145–175°
    [2] hip_angle_left    : 155–185°
    [3] hip_angle_right   : 155–185°
    [4] torso_lean        : -5–15° (앞 기울기 양수)
    [5] arm_swing_left    : 75–105°
    [6] arm_swing_right   : 75–105°
    [7] cadence           : 155–185 spm
    """
    rng = np.random.default_rng(42)
    means = np.array([160.0, 160.0, 170.0, 170.0, 5.0, 90.0, 90.0, 170.0])
    stds  = np.array([  8.0,   8.0,   8.0,   8.0, 4.0,  8.0,  8.0,   8.0])
    data = rng.normal(means, stds, size=(n, INPUT_DIM)).astype(np.float32)
    return data


class FeatureNormalizer:
    """Z-score 정규화 (학습 데이터 통계 기반)."""

    def __init__(self):
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, data: np.ndarray):
        self.mean = data.mean(axis=0)
        self.std = data.std(axis=0) + 1e-8

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        self.fit(data)
        return self.transform(data)


def train_autoencoder(
    model: RunningAutoencoder,
    normalizer: FeatureNormalizer,
    data: np.ndarray,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 256,
) -> float:
    """오토인코더 학습 후 최종 loss 반환."""
    normalized = normalizer.fit_transform(data)
    tensor = torch.from_numpy(normalized)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    dataset = torch.utils.data.TensorDataset(tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        for (batch,) in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch), batch)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        final_loss = criterion(model(tensor), tensor).item()
    return final_loss


def save_model(model: RunningAutoencoder, normalizer: FeatureNormalizer, path: str):
    torch.save(
        {
            "model": model.state_dict(),
            "mean": torch.from_numpy(normalizer.mean),
            "std": torch.from_numpy(normalizer.std),
        },
        path,
    )


def load_model(path: str) -> tuple[RunningAutoencoder, FeatureNormalizer]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = RunningAutoencoder()
    model.load_state_dict(ckpt["model"])
    model.eval()
    normalizer = FeatureNormalizer()
    normalizer.mean = ckpt["mean"].numpy()
    normalizer.std = ckpt["std"].numpy()
    return model, normalizer
