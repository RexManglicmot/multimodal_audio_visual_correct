# src/model.py

from pathlib import Path
from typing import Optional, Tuple

import torch
from torch import nn, Tensor
import torchvision.models as models
import yaml


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


class VideoBackbone(nn.Module):
    """
    ResNet18 backbone that outputs a 512-dim embedding per frame.
    Optionally frozen (no gradient updates).
    """

    def __init__(self, pretrained: bool = True, freeze: bool = True):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        # Replace final classification head with Identity so output is (B, 512)
        resnet.fc = nn.Identity()
        self.backbone = resnet

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B, 3, H, W)
        returns: (B, 512)
        """
        return self.backbone(x)


class AudioMLP(nn.Module):
    """
    Simple MLP that maps log-mel vector to an audio embedding.
    """

    def __init__(self, n_mels: int, h_audio: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_mels, h_audio),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B, n_mels)
        returns: (B, h_audio)
        """
        return self.net(x)


class MultiModalNet(nn.Module):
    """
    Multimodal network with:
      - Video branch: ResNet18 backbone -> 512-dim embedding -> optional linear head
      - Audio branch: MLP on log-mel features
      - Fusion: concat([video_emb, audio_emb]) -> classifier

    Modes:
      - "fusion": use both video + audio
      - "video_only": ignore audio branch
      - "audio_only": ignore video branch
    """

    def __init__(
        self,
        cfg: dict,
        mode: Optional[str] = None,
    ):
        super().__init__()

        self.cfg = cfg
        model_cfg = cfg["model"]
        video_cfg = cfg["video"]
        audio_cfg = cfg["audio"]

        self.num_classes = int(model_cfg["num_classes"])
        self.h_video = int(model_cfg["h_video"])
        self.h_audio = int(model_cfg["h_audio"])
        self.fusion_hidden = int(model_cfg["fusion_hidden"])
        self.dropout = float(model_cfg["dropout"])

        # Mode: "fusion" | "video_only" | "audio_only"
        self.mode = mode if mode is not None else model_cfg.get("mode", "fusion")
        assert self.mode in {"fusion", "video_only", "audio_only"}, f"Invalid mode: {self.mode}"

        # --- Video branch ---
        self.video_backbone = VideoBackbone(
            pretrained=bool(video_cfg.get("pretrained", True)),
            freeze=bool(video_cfg.get("freeze_backbone", True)),
        )

        # Map 512-dim backbone output -> h_video
        self.video_head = nn.Sequential(
            nn.Linear(512, self.h_video),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
        )

        # --- Audio branch ---
        n_mels = int(audio_cfg["n_mels"])
        self.audio_head = AudioMLP(
            n_mels=n_mels,
            h_audio=self.h_audio,
            dropout=self.dropout,
        )

        # --- Classifier (fusion or single-branch) ---
        if self.mode == "fusion":
            fusion_input_dim = self.h_video + self.h_audio
        elif self.mode == "video_only":
            fusion_input_dim = self.h_video
        else:  # "audio_only"
            fusion_input_dim = self.h_audio

        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, self.fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(self.fusion_hidden, self.num_classes),
        )

    def forward(
        self,
        video_frames: Optional[Tensor] = None,
        audio_feats: Optional[Tensor] = None,
    ) -> Tensor:
        """
        video_frames: (B, 3, H, W)
        audio_feats: (B, n_mels)
        returns: logits (B, num_classes)
        """
        batch_size = None
        video_emb = None
        audio_emb = None

        # --- Video branch ---
        if self.mode in {"fusion", "video_only"}:
            if video_frames is None:
                raise ValueError(f"video_frames cannot be None in mode={self.mode}")
            batch_size = video_frames.size(0)
            video_backbone_out = self.video_backbone(video_frames)  # (B, 512)
            video_emb = self.video_head(video_backbone_out)         # (B, h_video)

        # --- Audio branch ---
        if self.mode in {"fusion", "audio_only"}:
            if audio_feats is None:
                raise ValueError(f"audio_feats cannot be None in mode={self.mode}")
            if batch_size is None:
                batch_size = audio_feats.size(0)
            audio_emb = self.audio_head(audio_feats)                # (B, h_audio)

        # --- Fusion ---
        if self.mode == "fusion":
            fused = torch.cat([video_emb, audio_emb], dim=1)        # (B, h_video + h_audio)
        elif self.mode == "video_only":
            fused = video_emb                                       # (B, h_video)
        else:  # "audio_only"
            fused = audio_emb                                       # (B, h_audio)

        logits = self.classifier(fused)
        return logits


def build_model(
    config_path: str = "configs/config.yaml",
    mode: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> Tuple[MultiModalNet, dict]:
    """
    Convenience helper to:
      - load config
      - construct MultiModalNet
      - move it to device
    """
    cfg = load_config(config_path)
    if mode is None:
        mode = cfg["model"].get("mode", "fusion")

    model = MultiModalNet(cfg, mode=mode)

    if device is None:
        if cfg["train"]["device"] == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(cfg["train"]["device"])

    model = model.to(device)
    return model, cfg
