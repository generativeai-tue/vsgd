import math
import os
import time

import torch
import wandb

from utils.tester import test


def save_chpt(args, iteration, model, optimizer, scheduler, loss, name="last_chpt"):
    chpt = {
        "iteration": iteration,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
        "loss": loss,
    }
    torch.save(chpt, os.path.join(wandb.run.dir, f"{name}.pth"))
    wandb.save(os.path.join(wandb.run.dir, f"{name}.pth"), base_path=wandb.run.dir)
    print("->model saved<-\n")


def train(
    args,
    train_loader,
    val_loader,
    test_loader,
    model,
    optimizer,
    scheduler,
):
    with torch.no_grad():
        if val_loader is not None:
            # compute metrics on initialization
            batch = next(val_loader)
            history_val = run_iter(
                args=args,
                iteration=args.start_iter,
                batch=batch,
                model=model,
                optimizer=None,
                mode="val",
            )
            wandb.log({**history_val, "iter": args.start_iter})

    for iteration in range(args.start_iter, args.max_iter):
        batch = next(train_loader)

        time_start = time.time()
        history_train = run_iter(
            args,
            iteration=iteration,
            batch=batch,
            model=model,
            optimizer=optimizer,
            mode="train",
        )

        train_elapsed = time.time() - time_start
        time_start = time.time()
        history_val = {}

        if val_loader is not None:
            batch = next(val_loader)
            with torch.no_grad():
                history_val = run_iter(
                    args,
                    iteration=iteration + 1,
                    batch=batch,
                    model=model,
                    optimizer=None,
                    mode="val",
                )

            if scheduler is not None:
                if scheduler.__class__.__name__ == "ReduceLROnPlateau":
                    scheduler.step(history_val["val/loss"])
                else:
                    scheduler.step()

        val_elapsed = time.time() - time_start
        hist = {
            **history_train,
            **history_val,
            "train_time": train_elapsed,
            "val_time": val_elapsed,
        }

        # save metrics to wandb
        wandb.log(hist)
        # save checkpoint
        if iteration % args.save_freq == 0 or iteration == args.max_iter:
            loss = hist["train/loss"]
            if "val/loss" in hist.keys():
                loss = hist["val/loss"]
            save_chpt(
                args,
                iteration,
                model,
                optimizer,
                scheduler,
                loss,
            )

        if iteration % 100 == 0:
            print(
                "Iteration: {}/{}, Time elapsed: {:.2f}s\n"
                "* Train loss: {:.2f} \n".format(
                    iteration + 1,
                    args.max_iter,
                    val_elapsed + train_elapsed,
                    hist["train/loss"],
                )
            )
        if "val/loss" in hist.keys():
            if math.isnan(hist["val/loss"]):
                print("Nan loss, stopping training")
                break

        # run test eval to track the performance
        if (iteration + 1) % args.eval_test_freq == 0 and (
            iteration + 1
        ) < args.max_iter:
            print("Run test evaluation...")
            with torch.no_grad():
                test(
                    args=args,
                    loader=test_loader,
                    model=model,
                )

    print("Save last checkpoint")
    loss = hist["train/loss"]
    if "val/loss" in hist.keys():
        loss = hist["val/loss"]
    save_chpt(args, args.max_iter, model, optimizer, scheduler, loss)


def run_iter(args, iteration, batch, model, optimizer, mode="train"):
    if mode == "train":
        model.train()
        try:
            lr = optimizer.param_groups[0]["lr"]
        except:
            lr = 0.0
        history = {"lr": lr, "iter": iteration + 1}
        if iteration > 0:
            if args.optimizer_log_freq > 0 and iteration % args.optimizer_log_freq == 0:
                if hasattr(optimizer, "get_current_beta1_estimate"):
                    vals = optimizer.get_current_beta1_estimate()
                    vals = torch.cat([x.reshape(-1) for x in vals]).reshape(1, -1).cpu()
                    history["beta1"] = wandb.Histogram(vals)
                    history["beta1_median"] = vals.median()
                    history["beta1_mean"] = vals.mean()
                elif "betas" in optimizer.param_groups[0]:
                    beta1 = optimizer.param_groups[0]["betas"][0]
                    bias_correction = 1 - beta1**iteration
                    history["beta1_median"] = beta1 / bias_correction
                    history["beta1_mean"] = beta1 / bias_correction
                elif "momentum" in optimizer.param_groups[0]:
                    beta1 = optimizer.param_groups[0]["momentum"]
                    history["beta1_median"] = beta1
                    history["beta1_mean"] = beta1

    elif mode == "val":
        model.eval()
        history = {}

    if "cuda" in args.device:
        for i in range(len(batch)):
            batch[i] = batch[i].cuda(non_blocking=True)
    # Loss
    logs = {}
    if mode == "train":
        loss, logs = model.train_step(batch, device=args.device)
    elif mode == "val":
        with torch.no_grad():
            loss, logs = model.train_step(batch, device=args.device)

    if mode == "train":
        optimize(args, loss, model, optimizer)

    # Get the history
    for k in logs.keys():
        h_key = k
        if "/" not in k:
            h_key = f"{mode}/{k}"
        if "hist" in k:
            history[h_key] = wandb.Histogram(logs[k])
        else:
            history[h_key] = logs[k]

    return history


def optim_step(params, optimizer, grad_clip_val, grad_skip_val):
    # clip gradient
    grad_norm = torch.nn.utils.clip_grad_norm_(params, grad_clip_val).item()

    if grad_skip_val == 0 or grad_norm < grad_skip_val:
        optimizer.step()
    return grad_norm


def optimize(args, loss, model, optimizer):
    if args.grad_clip > 0:
        clip_to = args.grad_clip
    else:
        clip_to = 1e6

    logs = {"skipped_steps": 1}
    nans = torch.isnan(loss).sum().item()
    if nans == 0:
        logs["skipped_steps"] = 0
        # backprop through the main loss
        optimizer.zero_grad()
        loss.backward()
        params = [p for n, p in model.named_parameters() if p.requires_grad]

        grad_norm = optim_step(
            params=params,
            optimizer=optimizer,
            grad_clip_val=clip_to,
            grad_skip_val=args.grad_skip_thr,
        )
        logs["grad_norm"] = grad_norm

    wandb.log(logs)
