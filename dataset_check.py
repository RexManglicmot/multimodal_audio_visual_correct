from src.dataset import EgoFallsDataset

ds = EgoFallsDataset(
    metadata_path="data/processed/metadata_2k.csv",
    config_path="configs/config.yaml",
)

print(len(ds))          # should be 350
frame, audio_feat, label = ds[0]
print(frame.shape)      # torch.Size([3, 224, 224])   (given image_size=224)
print(audio_feat.shape) # torch.Size([64])           (given n_mels=64)
print(label)            # 0 or 1

"""
Yesss 🎉 that output is exactly what we want:

350 samples ✅

Frame shape: torch.Size([3, 224, 224]) ✅

Audio feature: torch.Size([64]) ✅

"""