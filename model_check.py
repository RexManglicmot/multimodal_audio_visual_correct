# model_check.py

import torch
from src.dataset import EgoFallsDataset
from src.model import build_model

# load a tiny batch from the dataset
ds = EgoFallsDataset("data/processed/metadata_2k.csv", "configs/config.yaml")
frame, audio_feat, label = ds[0]
frame = frame.unsqueeze(0)        # (1, 3, 224, 224)
audio_feat = audio_feat.unsqueeze(0)  # (1, 64)

model, cfg = build_model("configs/config.yaml", mode="fusion")
model.eval()

with torch.no_grad():
    logits = model(video_frames=frame, audio_feats=audio_feat)

print("logits shape:", logits.shape)  # expect (1, 2)
print("logits:", logits)
