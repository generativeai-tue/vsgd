import os

import torch
import wandb


def get_checkpoint(entity: str, project: str, idx: str, device: str = "cpu"):
    # download the checkpoint from wandb to the local machine.
    file = wandb.restore(
        "last_chpt.pth", run_path=os.path.join(entity, project, idx), replace=True
    )
    # load the checkpoint
    chpt = torch.load(file.name, map_location=device)
    return chpt
