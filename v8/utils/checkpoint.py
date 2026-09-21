import torch
from pathlib import Path


def save_checkpoint(model, run_name, epoch, cfg):
    save_dir = Path.home() / ".stable-wm" / "checkpoints" / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"weights_epoch_{epoch}.pt"
    torch.save(model.state_dict(), path)
    print(f"[Checkpoint] Saved to {path}")


def load_checkpoint(path, model):
    state_dict = torch.load(path, map_location="cpu")
    model.load_state_dict(state_dict)
    return model