import torch
import wandb
from tqdm import tqdm


def test(args, loader, model):
    model.eval()
    history = {}
    N = 0
    with torch.no_grad():
        for _, batch in tqdm(enumerate(loader)):
            if "cuda" in args.device:
                for i in range(len(batch)):
                    batch[i] = batch[i].cuda(non_blocking=True)

            N += batch[0].shape[0]
            logs = model.test_step(
                batch=batch,
            )

            for k in logs.keys():
                if f"test/{k}" not in history.keys():
                    history[f"test/{k}"] = 0.0
                history[f"test/{k}"] += logs[k]

        for k in history.keys():
            history[k] /= len(loader.dataset)

        wandb.log(history)
