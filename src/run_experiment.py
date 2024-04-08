import os
from pprint import pprint

import hydra.utils
import numpy as np
import omegaconf
import torch
import wandb
from hydra.utils import instantiate

import utils.tester as tester
import utils.trainer as trainer
from utils.wandb import get_checkpoint


def params_to(param, device):
    param.data = param.data.to(device)
    if param._grad is not None:
        param._grad.data = param._grad.data.to(device)


def optimizer_to(optim, device):
    for param in optim.state.values():
        if isinstance(param, torch.Tensor):
            params_to(param, device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    params_to(subparam, device)


def load_from_checkpoint(args, model, optimizer=None, scheduler=None):
    chpt = get_checkpoint(
        args.wandb.setup.entity,
        args.wandb.setup.project,
        args.train.resume_id,
        device="cpu",
    )
    args.train.start_iter = chpt["iteration"]
    # Load model and ema model
    model.load_state_dict(chpt["model_state_dict"])

    # Load optimizer
    if optimizer is not None:
        opt_state_dict = chpt["optimizer_state_dict"]
        optimizer.load_state_dict(opt_state_dict)
        optimizer_to(optimizer, args.train.device)

    # Load scheduler
    if scheduler is not None:
        scheduler_state_dict = chpt["scheduler_state_dict"]
        scheduler.load_state_dict(scheduler_state_dict)

    return args, model, optimizer, scheduler


def init_wandb(args):
    wandb.require("service")

    tags = [
        args.dataset.name,
        args.model.name,
        args.train.optimizer._target_,
        args.train.experiment_name,
    ]
    if args.train.resume_id is not None:
        wandb.init(
            **args.wandb.setup,
            id=args.train.resume_id,
            resume="must",
            settings=wandb.Settings(start_method="thread"),
        )
    else:
        wandb_cfg = omegaconf.OmegaConf.to_container(
            args, resolve=True, throw_on_missing=True
        )
        wandb.init(
            **args.wandb.setup,
            config=wandb_cfg,
            group=f"{args.model.name}_{args.dataset.name}"
            if args.wandb.group is None
            else args.wandb.group,
            tags=tags,
            dir=hydra.utils.get_original_cwd(),
            settings=wandb.Settings(start_method="thread"),
        )
    pprint(wandb.run.config)
    # define our custom x axis metric
    wandb.define_metric("iter")
    for pref in ["train", "val", "pic"]:
        wandb.define_metric(f"{pref}/*", step_metric="iter")
    wandb.define_metric("val/loss", summary="min", step_metric="iter")
    wandb.define_metric("test/loss", summary="min", step_metric="iter")


def compute_params(model, args):
    # add network size
    num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(num_param)
    wandb.run.summary["num_parameters"] = num_param


@hydra.main(version_base="1.3", config_path="../configs", config_name="defaults.yaml")
def run(args: omegaconf.DictConfig) -> None:
    # set cuda visible devices
    if args.train.device[-1] == "0":
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        args.train.device = "cuda"
    elif args.train.device[-1] == "1":
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        args.train.device = "cuda"

    # Set the seed
    torch.manual_seed(args.train.seed)
    torch.cuda.manual_seed(args.train.seed)
    np.random.seed(args.train.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ------------
    # data
    # ------------
    dset_params = {"root": os.path.join(hydra.utils.get_original_cwd(), "data/")}
    data_module = instantiate(args.dataset.data_module, **dset_params)
    data_module.setup()
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    # ------------
    # model & optimizer
    # ------------
    model = instantiate(args.model)
    optimizer = instantiate(args.train.optimizer, params=model.parameters())
    scheduler = None
    if hasattr(args.train, "scheduler"):
        scheduler = instantiate(args.train.scheduler, optimizer=optimizer)

    if args.train.resume_id is not None:
        print(f"Resume training {args.train.resume_id}")
        args, model, optimizer, scheduler = load_from_checkpoint(
            args, model, optimizer, scheduler
        )

    model.train()
    model.to(args.train.device)

    # ------------
    # logging
    # ------------
    init_wandb(args)
    wandb.watch(model, **args.wandb.watch)
    compute_params(model, args)

    # ------------
    # training
    # ------------
    if args.train.start_iter < args.train.max_iter:
        trainer.train(
            args.train,
            train_loader,
            val_loader,
            test_loader,
            model,
            optimizer,
            scheduler,
        )

    # ------------
    # testing
    # ------------
    model = instantiate(args.model)
    with omegaconf.open_dict(args):
        args.train.resume_id = wandb.run.id
    _, model, _, _ = load_from_checkpoint(args, model)
    model.to(args.train.device)

    tester.test(
        args.train,
        test_loader,
        model,
    )
    print("Test finished")
    wandb.finish()


if __name__ == "__main__":
    run()
