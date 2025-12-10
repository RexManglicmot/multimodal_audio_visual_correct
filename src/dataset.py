# src/dataset.py

from pathlib import Path
from typing import Optional, Sequence, Tuple

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as F_v
from torchvision.io import read_video
import torchaudio
import yaml


class EgoFallsDataset(Dataset):
    """
    EGOFALLS dataset that returns:
        - video frame tensor: (3, H, W)
        - audio log-mel feature vector: (n_mels,)
        - label: int {0, 1}
    """

    def __init__(
        self,
        metadata_path: str,
        config_path: str = "configs/config.yaml",
        indices: Optional[Sequence[int]] = None,
    ):
        super().__init__()

        # --- Load config ---
        self.cfg = self._load_config(config_path)
        self.paths_cfg = self.cfg["paths"]
        self.data_cfg = self.cfg["data"]
        self.video_cfg = self.cfg["video"]
        self.audio_cfg = self.cfg["audio"]

        # --- Load metadata ---
        metadata_path = Path(metadata_path)
        if not metadata_path.exists():
            raise FileNotFoundError(f"metadata_path not found: {metadata_path}")

        df = pd.read_csv(metadata_path)
        if indices is not None:
            df = df.iloc[list(indices)].reset_index(drop=True)

        self.df = df
        self.data_root = Path(self.paths_cfg["data_root"])
        self.path_col = self.data_cfg["path_col"]
        self.label_col = self.data_cfg["label_col"]

        # --- Video transform (single representative frame) ---
        self.img_size = self.video_cfg["image_size"]
        self.mean = self.video_cfg["mean"]
        self.std = self.video_cfg["std"]

        # These transforms operate directly on CHW tensors (no PIL needed)
        self.video_transform = T.Compose(
            [
                T.Resize((self.img_size, self.img_size)),  # for CHW tensors
                T.CenterCrop(self.img_size),
            ]
        )

        # --- Audio config ---
        self.target_sr = int(self.audio_cfg["sample_rate"])
        self.target_duration_sec = float(self.audio_cfg["target_duration_sec"])
        self.target_num_samples = int(self.target_sr * self.target_duration_sec)

        self.n_mels = int(self.audio_cfg["n_mels"])
        self.n_fft = int(self.audio_cfg["n_fft"])
        self.hop_length = int(self.audio_cfg["hop_length"])
        self.f_min = float(self.audio_cfg["f_min"])
        self.f_max = float(self.audio_cfg["f_max"])
        self.log_mel = bool(self.audio_cfg.get("log_mel", True))
        self.normalize_per_feature = bool(
            self.audio_cfg.get("normalize_per_feature", True)
        )

    @staticmethod
    def _load_config(config_path: str) -> dict:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        return cfg

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, int]:
        row = self.df.iloc[idx]

        rel_path = row[self.path_col]
        label = int(row[self.label_col])

        video_path = self.data_root / rel_path
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # pts_unit="sec" so info["audio_fps"] is Hz
        video, audio, info = read_video(str(video_path), pts_unit="sec")

        frame_tensor = self._process_video(video)
        audio_feat = self._process_audio(audio, info)

        return frame_tensor, audio_feat, label

    # ------------------------------------------------------------------
    # Video processing
    # ------------------------------------------------------------------
    def _process_video(self, video: Tensor) -> Tensor:
        """
        video: typically (T, C, H, W) or (T, H, W, C) depending on backend.
        Returns: transformed frame tensor (3, H, W), float32
        """
        if video.numel() == 0:
            return torch.zeros(3, self.img_size, self.img_size, dtype=torch.float32)

        t_dim = video.shape[0]
        mid_idx = t_dim // 2
        frame = video[mid_idx]  # 3D tensor

        # Normalize shape to CHW
        if frame.ndim != 3:
            raise ValueError(f"Unexpected frame ndim: {frame.ndim}, shape={frame.shape}")

        # If last dim looks like channels (1/3/4), assume HWC and permute
        if frame.shape[-1] in (1, 3, 4) and frame.shape[0] not in (1, 3, 4):
            # HWC -> CHW
            frame = frame.permute(2, 0, 1)

        # Now we expect CHW
        if frame.shape[0] not in (1, 3, 4):
            raise ValueError(
                f"Could not interpret frame as CHW. Got shape={frame.shape}"
            )

        frame = frame.float() / 255.0  # scale to [0, 1]

        # Resize + crop
        frame = self.video_transform(frame)  # still CHW

        # Ensure 3 channels (if grayscale, repeat)
        if frame.shape[0] == 1:
            frame = frame.repeat(3, 1, 1)

        # Normalize with ImageNet stats
        frame = F_v.normalize(frame, mean=self.mean, std=self.std)

        return frame

    # ------------------------------------------------------------------
    # Audio processing
    # ------------------------------------------------------------------
    def _process_audio(self, audio: Tensor, info: dict) -> Tensor:
        """
        audio: (num_audio_frames, num_channels) or empty
        Returns: log-mel vector (n_mels,)
        """
        if audio.numel() == 0:
            return torch.zeros(self.n_mels, dtype=torch.float32)

        # (num_audio_frames, num_channels) -> (channels, time)
        if audio.ndim == 2:
            audio = audio.transpose(0, 1)

        audio = audio.float()

        orig_sr = info.get("audio_fps", self.target_sr)
        orig_sr = int(orig_sr) if orig_sr is not None else self.target_sr

        if orig_sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(
                orig_freq=orig_sr, new_freq=self.target_sr
            )
            audio = resampler(audio)

        # mono: (1, time)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        elif audio.ndim == 1:
            audio = audio.unsqueeze(0)

        audio = self._pad_or_trim(audio, self.target_num_samples)

        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.target_sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_min=self.f_min,
            f_max=self.f_max,
        )

        mel = mel_transform(audio)  # (1, n_mels, time)
        mel = mel.squeeze(0)        # (n_mels, time)

        if self.log_mel:
            mel = torch.log(mel + 1e-6)

        feat = mel.mean(dim=-1)  # (n_mels,)

        if self.normalize_per_feature:
            m = feat.mean()
            s = feat.std()
            if s > 0:
                feat = (feat - m) / s

        return feat.to(torch.float32)

    @staticmethod
    def _pad_or_trim(audio: Tensor, target_len: int) -> Tensor:
        """
        audio: (1, time)
        """
        _, t = audio.shape
        if t == target_len:
            return audio
        if t > target_len:
            return audio[:, :target_len]

        pad_len = target_len - t
        pad = torch.zeros(1, pad_len, dtype=audio.dtype, device=audio.device)
        return torch.cat([audio, pad], dim=1)
