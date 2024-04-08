import torch
import torch.nn as nn


class ClassifierWrapper(nn.Module):
    def __init__(self, backbone, loss_fn=nn.CrossEntropyLoss(), **kwargs):
        super().__init__()
        self.backbone = backbone
        self.loss_fn = loss_fn

    def forward(self, batch):
        output = self.backbone(pixel_values=batch[0], return_dict=False)
        return output[0]

    def train_step(self, batch, scaler=None, device=None):
        if scaler is not None:
            with torch.autocast(device_type=device, dtype=torch.float16):
                logits = self.forward(batch)
                loss = self.loss_fn(logits, batch[1])
        else:
            logits = self.forward(batch)
            loss = self.loss_fn(logits, batch[1])

        logs = {
            "loss": loss.data,
            "accuracy": (logits.argmax(dim=1) == batch[1]).float().mean(),
        }
        return loss, logs

    def test_step(self, batch):
        logits = self.forward(batch)
        loss = self.loss_fn(logits, batch[1])

        logs = {
            "loss": loss.data * batch[0].shape[0],
            "accuracy": (logits.argmax(dim=1) == batch[1]).float().sum(),
        }
        return logs
