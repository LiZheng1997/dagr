# avoid matlab error on server
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

import torch
import tqdm
import wandb
from pathlib import Path
import argparse

from torch_geometric.data import DataLoader

from dagr.utils.logging import Checkpointer, set_up_logging_directory, log_hparams
from dagr.utils.buffers import DetectionBuffer
from dagr.utils.args import FLAGS
from dagr.utils.learning_rate_scheduler import LRSchedule

from dagr.data.augment import Augmentations
from dagr.utils.buffers import format_data
from dagr.data.dsec_data import DSEC

from dagr.model.networks.dagr import DAGR
from dagr.model.networks.ema import ModelEMA
from dagr.model.adaptive_sampler import AdaptiveEventSampler, AdaptiveSamplerConfig


def gradients_broken(model):
    valid_gradients = True
    for name, param in model.named_parameters():
        if param.grad is not None:
            # valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
            valid_gradients = not (torch.isnan(param.grad).any())
            if not valid_gradients:
                break
    return not valid_gradients

def fix_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            param.grad = torch.nan_to_num(param.grad, nan=0.0)


class EventSelector:
    def __init__(self, mode: str, budget: int,
                 candidate_budget: int = -1,
                 sampler: AdaptiveEventSampler = None):
        self.mode = mode
        self.budget = budget
        self.candidate_budget = candidate_budget
        self.sampler = sampler

    @property
    def enabled(self):
        return self.mode != "none" and self.budget > 0

    def _latest_indices(self, local_indices):
        if local_indices.numel() <= self.budget:
            return local_indices
        return local_indices[-self.budget:]

    def _adaptive_indices(self, data, local_indices):
        if (self.candidate_budget > self.budget
                and local_indices.numel() > self.candidate_budget):
            local_indices = local_indices[-self.candidate_budget:]

        if local_indices.numel() <= self.budget:
            return local_indices

        events = {
            "x": data.pos[local_indices, 0].long(),
            "y": data.pos[local_indices, 1].long(),
            "t": data.t[local_indices].float() / float(data.time_window[0]),
            "p": data.x[local_indices].view(-1).long(),
        }
        score_map = self.sampler.compute_score_map(events, None)
        scores = self.sampler._per_event_score(events, score_map)
        keep_local = scores.topk(self.budget).indices.sort().values
        return local_indices[keep_local]

    def __call__(self, data):
        if not self.enabled:
            return data, None

        keep_indices = []
        before = int(data.x.shape[0])
        per_graph_after = []
        num_graphs = int(data.num_graphs)

        with torch.no_grad():
            for batch_idx in range(num_graphs):
                local_indices = torch.nonzero(
                    data.batch == batch_idx, as_tuple=False
                ).view(-1)
                if self.mode == "adaptive":
                    selected = self._adaptive_indices(data, local_indices)
                else:
                    selected = self._latest_indices(local_indices)
                keep_indices.append(selected)
                per_graph_after.append(int(selected.numel()))

        if not keep_indices:
            return data, None

        keep = torch.cat(keep_indices).sort().values
        data.x = data.x[keep]
        data.pos = data.pos[keep]
        if hasattr(data, "t") and data.t is not None:
            data.t = data.t[keep]
        if hasattr(data, "batch") and data.batch is not None:
            data.batch = data.batch[keep]
        if hasattr(data, "ptr"):
            ptr = [0]
            for count in per_graph_after:
                ptr.append(ptr[-1] + count)
            data.ptr = torch.tensor(ptr, dtype=torch.long, device=data.x.device)
            if hasattr(data, "_slice_dict"):
                for key in ("x", "pos", "t"):
                    if key in data._slice_dict:
                        data._slice_dict[key] = data.ptr

        stats = {
            "before": before,
            "after": int(data.x.shape[0]),
            "fraction": float(data.x.shape[0]) / max(1, before),
        }
        return data, stats


def build_event_selector(args, height: int, width: int, device: str):
    mode = getattr(args, "event_selection", "none")
    budget = getattr(args, "event_budget", -1)
    if mode == "none" or budget <= 0:
        return EventSelector("none", -1)

    candidate_budget = getattr(args, "event_candidate_budget", -1)
    sampler = None
    if mode == "adaptive":
        if args.adaptive_sampler_checkpoint is None:
            raise ValueError(
                "--adaptive_sampler_checkpoint is required for adaptive selection"
            )
        ckpt = torch.load(args.adaptive_sampler_checkpoint, map_location=device)
        cfg = AdaptiveSamplerConfig(**ckpt["cfg"])
        if cfg.sensor_h != height or cfg.sensor_w != width:
            raise ValueError(
                f"Sampler size {(cfg.sensor_h, cfg.sensor_w)} does not match "
                f"dataset size {(height, width)}"
            )
        sampler = AdaptiveEventSampler(cfg).to(device).eval()
        sampler.load_state_dict(ckpt["state_dict"])
        for param in sampler.parameters():
            param.requires_grad_(False)
        print(
            "[event-selection] loaded adaptive sampler "
            f"{args.adaptive_sampler_checkpoint}"
        )

    if candidate_budget > 0:
        print(f"[event-selection] candidate_budget={candidate_budget}")
    print(f"[event-selection] mode={mode} budget={budget}")
    return EventSelector(mode, budget, candidate_budget, sampler)


def train(loader: DataLoader,
          model: torch.nn.Module,
          ema: ModelEMA,
          scheduler: torch.optim.lr_scheduler.LambdaLR,
          optimizer: torch.optim.Optimizer,
          args: argparse.ArgumentParser,
          event_selector: EventSelector = None,
          run_name=""):

    model.train()

    for i, data in enumerate(tqdm.tqdm(loader, desc=f"Training {run_name}")):
        data = data.cuda(non_blocking=True)
        if event_selector is not None:
            data, selection_stats = event_selector(data)
        else:
            selection_stats = None
        data = format_data(data)

        optimizer.zero_grad(set_to_none=True)

        model_outputs = model(data)

        loss_dict = {k: v for k, v in model_outputs.items() if "loss" in k}
        loss = loss_dict.pop("total_loss")

        loss.backward()

        torch.nn.utils.clip_grad_value_(model.parameters(), args.clip)

        fix_gradients(model)

        optimizer.step()
        scheduler.step()

        ema.update(model)

        training_logs = {f"training/loss/{k}": v for k, v in loss_dict.items()}
        logs = {
            "training/loss": loss.item(),
            "training/lr": scheduler.get_last_lr()[-1],
            **training_logs,
        }
        if selection_stats is not None:
            logs.update({
                "training/event_selection/before": selection_stats["before"],
                "training/event_selection/after": selection_stats["after"],
                "training/event_selection/fraction": selection_stats["fraction"],
            })
        wandb.log(logs)
        if i == 0 or (i + 1) % 100 == 0:
            status = (
                f"[train] step={i + 1} loss={loss.item():.6f} "
                f"lr={scheduler.get_last_lr()[-1]:.6e}"
            )
            if selection_stats is not None:
                status += (
                    " event_selection="
                    f"{selection_stats['after']}/{selection_stats['before']} "
                    f"({selection_stats['fraction']:.4f})"
                )
            print(status, flush=True)

        if args.max_train_steps > 0 and i + 1 >= args.max_train_steps:
            break

def run_test(loader: DataLoader,
         model: torch.nn.Module,
         event_selector: EventSelector = None,
         dry_run_steps: int=-1,
         dataset="gen1"):

    model.eval()

    mapcalc = DetectionBuffer(height=loader.dataset.height, width=loader.dataset.width, classes=loader.dataset.classes)

    for i, data in enumerate(tqdm.tqdm(loader)):
        data = data.cuda()
        if event_selector is not None:
            data, _ = event_selector(data)
        data = format_data(data)

        detections, targets = model(data)
        if i % 10 == 0:
            torch.cuda.empty_cache()

        mapcalc.update(detections, targets, dataset, data.height[0], data.width[0])

        if dry_run_steps > 0 and i == dry_run_steps:
            break

    torch.cuda.empty_cache()

    return mapcalc

if __name__ == '__main__':
    import torch_geometric
    import random
    import numpy as np

    args = FLAGS()

    seed = args.seed
    torch_geometric.seed.seed_everything(seed)
    torch.random.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    output_directory = set_up_logging_directory(args.dataset, args.task, args.output_directory, exp_name=args.exp_name)
    log_hparams(args)

    augmentations = Augmentations(args)

    print("init datasets")
    dataset_path = args.dataset_directory / args.dataset

    train_dataset = DSEC(root=dataset_path, split="train", transform=augmentations.transform_training, debug=False,
                         min_bbox_diag=15, min_bbox_height=10)
    test_dataset = DSEC(root=dataset_path, split="val", transform=augmentations.transform_testing, debug=False,
                        min_bbox_diag=15, min_bbox_height=10)

    train_loader = DataLoader(train_dataset, follow_batch=['bbox', 'bbox0'], batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    num_iters_per_epoch = len(train_loader)

    sampler = np.random.permutation(np.arange(len(test_dataset)))
    test_loader = DataLoader(test_dataset, sampler=sampler, follow_batch=['bbox', 'bbox0'], batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)

    print("init net")
    # load a dummy sample to get height, width
    model = DAGR(args, height=test_dataset.height, width=test_dataset.width)

    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print(f"Training with {num_params} number of parameters.")

    model = model.cuda()
    ema = ModelEMA(model)

    nominal_batch_size = 64
    lr = args.l_r * np.sqrt(args.batch_size) / np.sqrt(nominal_batch_size)
    optimizer = torch.optim.AdamW(list(model.parameters()), lr=lr, weight_decay=args.weight_decay)

    lr_func = LRSchedule(warmup_epochs=.3,
                         num_iters_per_epoch=num_iters_per_epoch,
                         tot_num_epochs=args.tot_num_epochs)

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lr_func)

    checkpointer = Checkpointer(output_directory=output_directory,
                                model=model, optimizer=optimizer,
                                scheduler=lr_scheduler, ema=ema,
                                args=args)

    start_epoch = checkpointer.restore_if_existing(output_directory, resume_from_best=False)

    start_epoch = 0
    if hasattr(args, "resume_checkpoint"):
        start_epoch = checkpointer.restore_checkpoint(args.resume_checkpoint, best=False)
        print(f"Resume from checkpoint at epoch {start_epoch}")

    event_selector = build_event_selector(
        args, height=test_dataset.height, width=test_dataset.width,
        device="cuda",
    )

    if not args.skip_initial_eval:
        with torch.no_grad():
            mapcalc = run_test(
                test_loader, ema.ema, event_selector=event_selector,
                dry_run_steps=args.dry_run_val_steps, dataset=args.dataset,
            )
        mapcalc.compute()

    print("starting to train")
    for epoch in range(start_epoch, args.tot_num_epochs):
        train(
            train_loader, model, ema, lr_scheduler, optimizer, args,
            event_selector=event_selector, run_name=wandb.run.name,
        )
        checkpointer.checkpoint(epoch, name=f"last_model")

        if args.skip_epoch_eval:
            continue

        if epoch % 3 > 0:
            continue

        with torch.no_grad():
            mapcalc = run_test(
                test_loader, ema.ema, event_selector=event_selector,
                dry_run_steps=args.dry_run_val_steps, dataset=args.dataset,
            )
            metrics = mapcalc.compute()
            checkpointer.process(metrics, epoch)
