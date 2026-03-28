from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from app.models.unet_light import UNetLight
from app.utils.dataset_utils import BratsSliceDataset


def train() -> None:
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # Lightweight subset only (10-20 samples) for CPU-friendly training
    dataset = BratsSliceDataset(size=(128, 128), limit=20)
    if len(dataset) < 10:
        print(f"Warning: found only {len(dataset)} samples; recommended 10-20.")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = UNetLight(in_channels=1, base=16).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 3
    print(f"Starting lightweight training for {epochs} epochs on {len(dataset)} samples...")
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())

        avg_loss = epoch_loss / max(1, len(dataloader))
        print(f"Epoch [{epoch + 1}/{epochs}] loss={avg_loss:.4f}")

    save_path = "unet_trained.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Saved trained weights to: {save_path}")


if __name__ == "__main__":
    train()
