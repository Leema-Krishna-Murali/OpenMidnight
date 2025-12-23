# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import gc
import logging
import os
import sys
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torchvision import transforms
from torchvision.datasets import folder
import torch.utils.data

from datasets import load_dataset
from PIL import Image, ImageOps
import pyarrow
import pyarrow.dataset
import torch.distributed as dist

import dinov2.distributed as distributed
from dinov2.data.augmentations_ablation import DataAugmentationDINO
from dinov2.fsdp import FSDPCheckpointer
from dinov2.logging import MetricLogger
from dinov2.models import build_model_from_cfg
from dinov2.train.jepa_logging import JEPALogger
from dinov2.utils.config import default_setup, write_config
from dinov2.utils.param_groups import get_params_groups_with_decay, fuse_params_groups

import wandb

sys.path.insert(0, "/home/paul/lejepa")
import lejepa

torch.backends.cuda.matmul.allow_tf32 = True
logger = logging.getLogger("dinov2")


def _load_cfg(config_path: str):
    from omegaconf import OmegaConf
    from dinov2.configs import dinov2_default_config

    cfg = OmegaConf.load(config_path)
    cfg = OmegaConf.merge(OmegaConf.create(dinov2_default_config), cfg)
    if not cfg.train.output_dir:
        raise ValueError("cfg.train.output_dir must be set")
    cfg.train.output_dir = os.path.abspath(cfg.train.output_dir)
    return cfg


def _setup(cfg, config_path: str):
    os.makedirs(cfg.train.output_dir, exist_ok=True)
    args = SimpleNamespace(output_dir=cfg.train.output_dir, seed=cfg.train.seed, config_file=config_path)
    default_setup(args)
    write_config(cfg, cfg.train.output_dir)
    return cfg


def _build_streaming_dataset(
    dataset_path: str,
    *,
    shuffle_buffer: int,
    base_seed: int,
    fragment_prefetch_limit: int,
    fragment_range_size: int,
    epoch: int,
):
    world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    global_rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

    fragment_scan_options = pyarrow.dataset.ParquetFragmentScanOptions(
        cache_options=pyarrow.CacheOptions(
            prefetch_limit=fragment_prefetch_limit,
            range_size_limit=fragment_range_size,
        ),
    )

    ds = load_dataset(
        dataset_path,
        streaming=True,
        fragment_scan_options=fragment_scan_options,
    )["train"]

    if world_size > 1:
        ds = ds.shard(num_shards=world_size, index=global_rank)

    seed = base_seed + epoch * 1_000_000 + global_rank * 10000
    ds = ds.shuffle(buffer_size=shuffle_buffer, seed=seed)
    return ds


def _collate_lejepa(samples_list, dtype):
    images = []
    indexes = []
    for sample in samples_list:
        images.append(sample[0])
        indexes.append(sample[1])

    n_global_crops = len(images[0]["global_crops"])
    n_local_crops = len(images[0]["local_crops"])

    collated_global_crops = torch.stack(
        [s["global_crops"][i] for i in range(n_global_crops) for s in images]
    ).to(dtype)

    if n_local_crops > 0:
        collated_local_crops = torch.stack(
            [s["local_crops"][i] for i in range(n_local_crops) for s in images]
        ).to(dtype)
    else:
        collated_local_crops = torch.empty((0,), dtype=dtype)

    return {
        "collated_global_crops": collated_global_crops,
        "collated_local_crops": collated_local_crops,
        "indexes": indexes,
    }


class ProjectionMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, nlayers, norm):
        super().__init__()
        if nlayers < 1:
            raise ValueError("proj_nlayers must be >= 1")

        layers = []
        dim = in_dim
        for _ in range(nlayers - 1):
            layers.append(nn.Linear(dim, hidden_dim, bias=False))
            if norm == "batchnorm":
                layers.append(nn.BatchNorm1d(hidden_dim))
            elif norm == "layernorm":
                layers.append(nn.LayerNorm(hidden_dim))
            else:
                raise ValueError("proj_norm must be 'batchnorm' or 'layernorm'")
            layers.append(nn.GELU())
            dim = hidden_dim
        layers.append(nn.Linear(dim, out_dim, bias=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class LeJEPAModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        student_backbone, _, embed_dim = build_model_from_cfg(cfg)
        self.backbone = student_backbone
        self.projector = ProjectionMLP(
            embed_dim,
            cfg.lejepa.proj_hidden_dim,
            cfg.lejepa.proj_dim,
            cfg.lejepa.proj_nlayers,
            cfg.lejepa.proj_norm,
        )
        self.embed_dim = embed_dim


def _load_pretrained_backbone(cfg, backbone):
    def _iter_vit_blocks(backbone):
        if backbone.chunked_blocks:
            for chunk in backbone.blocks:
                for blk in chunk:
                    if not isinstance(blk, torch.nn.Identity):
                        yield blk
        else:
            for blk in backbone.blocks:
                yield blk

    def _mlp_kind(block):
        if hasattr(block.mlp, "fc1"):
            return "mlp"
        if hasattr(block.mlp, "w12"):
            return "swiglu"
        raise AssertionError("Unsupported FFN block type")

    logger.info("Loading pretrained backbone from torch.hub: %s", cfg.train.pretrained_name)
    model_pretrained = torch.hub.load(cfg.train.pretrained_repo, cfg.train.pretrained_name)
    device = next(backbone.parameters()).device
    model_pretrained = model_pretrained.to(device)

    with torch.no_grad():
        if backbone.embed_dim != model_pretrained.embed_dim:
            raise AssertionError("Pretrained embed_dim mismatch")
        if backbone.n_blocks != model_pretrained.n_blocks:
            raise AssertionError("Pretrained depth mismatch")
        if backbone.num_register_tokens != model_pretrained.num_register_tokens:
            raise AssertionError("Pretrained register token count mismatch")

        backbone.patch_embed.proj.weight.copy_(model_pretrained.patch_embed.proj.weight)
        backbone.patch_embed.proj.bias.copy_(model_pretrained.patch_embed.proj.bias)
        backbone.cls_token.copy_(model_pretrained.cls_token)
        backbone.mask_token.copy_(model_pretrained.mask_token)
        if backbone.num_register_tokens:
            backbone.register_tokens.copy_(model_pretrained.register_tokens)

        pos_embed_pretrained = model_pretrained.pos_embed.detach()
        n_extra_tokens = 1
        cls_pos_embed = pos_embed_pretrained[:, :n_extra_tokens]
        patch_pos_embed = pos_embed_pretrained[:, n_extra_tokens:]

        orig_size = int(patch_pos_embed.shape[1] ** 0.5)
        patch_pos_embed = patch_pos_embed.reshape(1, orig_size, orig_size, -1).permute(0, 3, 1, 2)

        target_h, target_w = backbone.patch_embed.patches_resolution
        resized_patch_pos_embed = F.interpolate(
            patch_pos_embed,
            size=(target_h, target_w),
            mode="bicubic",
            align_corners=False,
            antialias=model_pretrained.interpolate_antialias,
        )
        resized_patch_pos_embed = resized_patch_pos_embed.permute(0, 2, 3, 1).reshape(
            1, target_h * target_w, -1
        )
        new_pos_embed = torch.cat((cls_pos_embed, resized_patch_pos_embed), dim=1)
        backbone.pos_embed.copy_(new_pos_embed)

        backbone_blocks = list(_iter_vit_blocks(backbone))
        pretrained_blocks = list(_iter_vit_blocks(model_pretrained))
        if len(backbone_blocks) != len(pretrained_blocks):
            raise AssertionError("Pretrained block count mismatch")
        if backbone_blocks and _mlp_kind(backbone_blocks[0]) != _mlp_kind(pretrained_blocks[0]):
            raise AssertionError(
                f"FFN mismatch: cfg.student.ffn_layer builds {_mlp_kind(backbone_blocks[0])}, "
                f"but torch.hub {cfg.train.pretrained_name} uses {_mlp_kind(pretrained_blocks[0])}"
            )
        for dst, src in zip(backbone_blocks, pretrained_blocks):
            dst.load_state_dict(src.state_dict(), strict=True)

        backbone.norm.weight.copy_(model_pretrained.norm.weight)
        backbone.norm.bias.copy_(model_pretrained.norm.bias)

    del model_pretrained
    gc.collect()


def _build_optimizer(cfg, model):
    param_groups = get_params_groups_with_decay(
        model=model,
        lr_decay_rate=cfg.optim.layerwise_decay,
        patch_embed_lr_mult=cfg.optim.patch_embed_lr_mult,
    )
    fused_params_groups = fuse_params_groups(param_groups)
    for g in fused_params_groups:
        g["foreach"] = True
        g["lr"] = cfg.optim.lr * g["lr_multiplier"]
        g["weight_decay"] = cfg.optim.weight_decay * g["wd_multiplier"]
    return torch.optim.AdamW(fused_params_groups, betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2))


def _build_lr_scheduler(cfg, optimizer):
    steps_per_epoch = cfg.train.OFFICIAL_EPOCH_LENGTH
    warmup_steps = cfg.optim.warmup_epochs * steps_per_epoch
    total_steps = cfg.optim.epochs * steps_per_epoch
    s1 = LinearLR(optimizer, start_factor=cfg.optim.warmup_factor, total_iters=warmup_steps)
    s2 = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=cfg.optim.eta_min)
    return SequentialLR(optimizer, schedulers=[s1, s2], milestones=[warmup_steps])


def _build_sigreg(cfg, device):
    univariate_test = lejepa.univariate.EppsPulley(
        t_max=cfg.lejepa.t_max,
        n_points=cfg.lejepa.n_points,
    )
    sigreg = lejepa.multivariate.SlicingUnivariateTest(
        univariate_test=univariate_test,
        num_slices=cfg.lejepa.num_slices,
    )
    return sigreg.to(device)


def _do_bach_eval(cfg, backbone, iteration):
    is_main = distributed.is_main_process()
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    if not is_main:
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        return

    bach_root = str(cfg.evaluation.bach_root)
    if not os.path.isdir(bach_root):
        logger.info("Skipping BACH eval; dataset path missing: %s", bach_root)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        return

    model = backbone.module if hasattr(backbone, "module") else backbone
    model.eval()
    model.requires_grad_(False)
    device = next(model.parameters()).device

    class _ResizeAndCrop(transforms.Compose):
        def __init__(self, size=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
            ops = [
                transforms.Resize(size),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
            super().__init__(ops)

    _BACH_TRAIN_INDEX_RANGES = [
        (0, 41),
        (59, 60),
        (90, 139),
        (169, 240),
        (258, 260),
        (273, 345),
        (368, 400),
    ]
    _BACH_VAL_INDEX_RANGES = [
        (41, 59),
        (60, 90),
        (139, 169),
        (240, 258),
        (260, 273),
        (345, 368),
    ]
    _BACH_CLASS_TO_IDX = {"Benign": 0, "InSitu": 1, "Invasive": 2, "Normal": 3}

    class _BACHDataset(torch.utils.data.Dataset):
        def __init__(self, root, split, transform):
            self.root = os.path.abspath(os.path.expanduser(root))
            self.split = split
            self.transform = transform
            dataset_path = os.path.join(self.root, "ICIAR2018_BACH_Challenge", "Photos")
            self.samples = folder.make_dataset(
                directory=dataset_path,
                class_to_idx=_BACH_CLASS_TO_IDX,
                extensions=(".tif",),
            )
            if len(self.samples) == 0:
                raise RuntimeError(f"No BACH images found in {dataset_path}")
            if split == "train":
                index_ranges = _BACH_TRAIN_INDEX_RANGES
            elif split == "val":
                index_ranges = _BACH_VAL_INDEX_RANGES
            else:
                raise ValueError("Invalid BACH split. Use 'train' or 'val'.")
            indices = []
            for start, end in index_ranges:
                indices.extend(range(start, end))
            self.indices = indices

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            image_path, target = self.samples[self.indices[idx]]
            image = Image.open(image_path).convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
            target_tensor = torch.tensor(target, dtype=torch.long)
            return image, target_tensor

    transform = _ResizeAndCrop(
        size=224,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    train_ds = _BACHDataset(root=bach_root, split="train", transform=transform)
    val_ds = _BACHDataset(root=bach_root, split="val", transform=transform)

    predict_batch_size = 64
    num_workers = 4

    def _compute_embeddings(dataset):
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=predict_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        feats = []
        targets = []
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                out = model(images, is_training=True)
                cls = out["x_norm_clstoken"]
                feats.append(cls)
                targets.append(labels)
        feats = torch.cat(feats, dim=0)
        targets = torch.cat(targets, dim=0)
        return feats, targets

    train_feats, train_targets = _compute_embeddings(train_ds)
    val_feats, val_targets = _compute_embeddings(val_ds)

    in_features = train_feats.shape[-1]
    num_classes = 4
    head = torch.nn.Linear(in_features, num_classes, bias=True).to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=1e-2)

    train_dataset = torch.utils.data.TensorDataset(train_feats, train_targets)
    train_batch_size = 256
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        drop_last=False,
    )

    val_dataset = torch.utils.data.TensorDataset(val_feats, val_targets)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=train_batch_size,
        shuffle=False,
        drop_last=False,
    )

    def _eval_head():
        head.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for feats_batch, targets_batch in val_loader:
                feats_batch = feats_batch.to(device, non_blocking=True)
                logits = head(feats_batch)
                preds = logits.argmax(dim=1).cpu()
                all_preds.append(preds)
                all_targets.append(targets_batch.cpu())
        preds = torch.cat(all_preds, dim=0)
        targets = torch.cat(all_targets, dim=0)
        plain_acc = float((preds == targets).float().mean().item())
        conf = torch.zeros(num_classes, num_classes, dtype=torch.long)
        indices = targets * num_classes + preds
        bincount = torch.bincount(indices, minlength=num_classes * num_classes)
        conf = bincount.view(num_classes, num_classes)
        per_class = conf.diag().float() / conf.sum(dim=1).clamp_min(1)
        balanced_acc = float(per_class.mean().item())
        head.train()
        return plain_acc, balanced_acc

    max_steps = 12500
    eval_every = 250
    patience = 1250
    steps = 0
    best_plain = -1.0
    best_balanced = -1.0
    best_state = None
    steps_since_improve = 0
    head.train()
    while steps < max_steps:
        for feats_batch, targets_batch in train_loader:
            feats_batch = feats_batch.to(device, non_blocking=True)
            targets_batch = targets_batch.to(device, non_blocking=True)
            logits = head(feats_batch)
            loss = criterion(logits, targets_batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            steps += 1
            if steps % eval_every == 0 or steps >= max_steps:
                plain_acc, balanced_acc = _eval_head()
                if plain_acc > best_plain:
                    best_plain = plain_acc
                    best_balanced = balanced_acc
                    best_state = {k: v.cpu() for k, v in head.state_dict().items()}
                    steps_since_improve = 0
                else:
                    steps_since_improve += eval_every
                if steps_since_improve >= patience:
                    steps = max_steps
                    break
            if steps >= max_steps:
                break

    if best_state is not None:
        head.load_state_dict(best_state)
        bach_acc_plain, bach_acc_balanced = best_plain, best_balanced
    else:
        bach_acc_plain, bach_acc_balanced = _eval_head()

    logger.info(
        "BACH val accuracy (linear probe): plain=%.4f balanced=%.4f",
        bach_acc_plain,
        bach_acc_balanced,
    )

    if wandb.run is not None and distributed.is_main_process():
        step = iteration if isinstance(iteration, int) else int(str(iteration).split("_")[-1])
        wandb.log(
            {
                "val/BACH_BALANCED_ACCURACY": bach_acc_balanced,
                "val/BACH_MULTICLASS_ACCURACY": bach_acc_plain,
            },
            step=step,
        )

    model.train()
    model.requires_grad_(True)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _do_train(cfg, model, sigreg):
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[cfg.train.amp_dtype]
    inputs_dtype = amp_dtype
    scaler = GradScaler(enabled=cfg.train.amp_dtype == "fp16")
    model_root = model.module if hasattr(model, "module") else model

    optimizer = _build_optimizer(cfg, model)
    scheduler = _build_lr_scheduler(cfg, optimizer)

    if distributed.is_main_process():
        from omegaconf import OmegaConf

        run_id_path = Path(cfg.train.output_dir) / "wandb_run_id.txt"
        if cfg.train.resume and run_id_path.exists():
            run_id = run_id_path.read_text().strip()
            resume_mode = "must"
        else:
            run_id_path.parent.mkdir(parents=True, exist_ok=True)
            run_id = wandb.util.generate_id()
            run_id_path.write_text(run_id)
            resume_mode = "allow"
        run = wandb.init(
            project=cfg.train.wandb_project,
            config=OmegaConf.to_container(cfg),
            name=cfg.train.wandb_name,
            id=run_id,
            resume=resume_mode,
        )
        artifact = wandb.Artifact(name=f"run-source-{run.id}", type="code")
        artifact.add_file(str(Path(__file__).resolve()))
        artifact.add_file(str(Path(cfg.train.output_dir) / "config.yaml"))
        run.log_artifact(artifact)

    if not cfg.train.skip_checkpointer:
        checkpointer = FSDPCheckpointer(model, cfg.train.output_dir, optimizer=optimizer, save_to_disk=True)
        start_iter = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=cfg.train.resume).get(
            "iteration", -1
        ) + 1
    else:
        start_iter = 0
        checkpointer = None
    if start_iter > 0:
        # Fast-forward in chainable form so resume aligns with the current iteration.
        for _ in range(start_iter):
            scheduler.step()

    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    max_iter = cfg.optim.epochs * OFFICIAL_EPOCH_LENGTH
    eta_target_iter = max_iter

    if not cfg.train.skip_checkpointer:
        from fvcore.common.checkpoint import PeriodicCheckpointer

        periodic_checkpointer = PeriodicCheckpointer(
            checkpointer,
            period=3 * OFFICIAL_EPOCH_LENGTH,
            max_iter=max_iter,
            max_to_keep=3,
        )
    else:
        periodic_checkpointer = None

    data_transform = DataAugmentationDINO(
        cfg.crops.global_crops_scale,
        cfg.crops.local_crops_scale,
        cfg.crops.local_crops_number,
        global_crops_size=cfg.crops.global_crops_size,
        local_crops_size=cfg.crops.local_crops_size,
    )

    dataset_builder = lambda epoch: _build_streaming_dataset(
        dataset_path=str(cfg.train.streaming_dataset_path),
        shuffle_buffer=cfg.train.shuffle_buffer,
        base_seed=cfg.train.seed,
        fragment_prefetch_limit=cfg.train.fragment_prefetch_limit,
        fragment_range_size=cfg.train.fragment_range_size,
        epoch=epoch,
    )

    def decode_and_transform(item):
        image = Image.open(BytesIO(item["image_bytes"]))
        image = ImageOps.exif_transpose(image).convert("RGB")
        transformed = data_transform(image)
        slide_meta = (item["slide_path"], item["x"], item["y"], item["level"])
        return transformed, slide_meta

    class _TransformedStreamingDataset(torch.utils.data.IterableDataset):
        def __init__(self, dataset_builder, transform, samples_per_epoch, reshuffle_every=0):
            self._dataset_builder = dataset_builder
            self._transform = transform
            self._samples_per_epoch = samples_per_epoch
            self._reshuffle_every = reshuffle_every
            self._initialized = False
            self._epoch_seen = 0
            self._src_iter = None

        def _init_or_reshuffle(self, *, force: bool = False):
            if force or (not self._initialized) or (
                self._reshuffle_every and (self._epoch_seen % self._reshuffle_every == 0)
            ):
                src = self._dataset_builder(epoch=self._epoch_seen if self._reshuffle_every else 0)
                worker_info = torch.utils.data.get_worker_info()
                if worker_info is not None and worker_info.num_workers > 1:
                    src = src.shard(num_shards=worker_info.num_workers, index=worker_info.id)
                self._src_iter = iter(src)
                self._initialized = True

        def __iter__(self):
            while True:
                self._init_or_reshuffle()

                rank_quota = self._samples_per_epoch
                worker_info = torch.utils.data.get_worker_info()
                num_workers = worker_info.num_workers if worker_info is not None else 1
                worker_id = worker_info.id if worker_info is not None else 0
                base = rank_quota // num_workers
                remainder = rank_quota % num_workers
                local_quota = base + (1 if worker_id < remainder else 0)

                produced = 0
                while produced < local_quota:
                    try:
                        sample = next(self._src_iter)
                    except StopIteration:
                        self._init_or_reshuffle(force=True)
                        continue
                    yield self._transform(sample)
                    produced += 1

                self._epoch_seen += 1

    samples_per_epoch = cfg.train.batch_size_per_gpu * cfg.train.OFFICIAL_EPOCH_LENGTH
    dataset = _TransformedStreamingDataset(
        dataset_builder,
        decode_and_transform,
        samples_per_epoch=samples_per_epoch,
    )

    def _worker_init(_):
        torch.set_num_threads(1)
        os.environ.setdefault("OMP_NUM_THREADS", "1")

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.train.batch_size_per_gpu,
        num_workers=cfg.train.num_workers,
        drop_last=True,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=lambda samples: _collate_lejepa(samples, inputs_dtype),
        prefetch_factor=4,
        worker_init_fn=_worker_init,
    )

    jepa_logger = JEPALogger(cfg)
    iteration = start_iter
    metrics_file = os.path.join(cfg.train.output_dir, "training_metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)
    header = "Training"

    for data in metric_logger.log_every(
        data_loader,
        10,
        header,
        eta_target_iter + 1,
        start_iter,
    ):
        iter_start, data_time, log_heavy = jepa_logger.start_iter(iteration)
        if iteration >= max_iter:
            logger.info("Stopping at iteration {}".format(iteration))
            if cfg.evaluation.eval_period_iterations >= 0:
                eval_backbone = model.module.backbone if hasattr(model, "module") else model.backbone
                _do_bach_eval(cfg, eval_backbone, f"training_{iteration}")
                torch.cuda.synchronize()
            if not cfg.train.skip_checkpointer and checkpointer is not None:
                checkpointer.save(f"model_{iteration:07d}", iteration=iteration)
            break

        if cfg.evaluation.eval_period_iterations >= 0 and iteration % cfg.evaluation.eval_period_iterations == 0:
            eval_backbone = model.module.backbone if hasattr(model, "module") else model.backbone
            _do_bach_eval(cfg, eval_backbone, f"training_{iteration}")
            torch.cuda.synchronize()

        if not cfg.train.skip_checkpointer and periodic_checkpointer is not None:
            periodic_checkpointer.step(iteration)

        lr = optimizer.param_groups[0]["lr"]
        wd = cfg.optim.weight_decay

        optimizer.zero_grad(set_to_none=True)

        n_global_crops = 2
        n_local_crops = cfg.crops.local_crops_number

        global_crops = data["collated_global_crops"].cuda(non_blocking=True)
        batch_size = global_crops.shape[0] // n_global_crops
        if n_local_crops > 0:
            local_crops = data["collated_local_crops"].cuda(non_blocking=True)
        else:
            local_crops = None

        with autocast(dtype=amp_dtype, enabled=True):
            global_out = model_root.backbone(global_crops, is_training=True)["x_norm_clstoken"]
            global_out = global_out.view(n_global_crops, batch_size, -1)
            if n_local_crops > 0:
                local_out = model_root.backbone(local_crops, is_training=True)["x_norm_clstoken"]
                local_out = local_out.view(n_local_crops, batch_size, -1)
                all_out = torch.cat([global_out, local_out], dim=0)
            else:
                all_out = global_out

            proj = model_root.projector(all_out.flatten(0, 1))
            proj = proj.view(all_out.shape[0], batch_size, -1).float()
            centers = proj[:n_global_crops].mean(dim=0)
            pred_loss = (centers - proj).square().mean()

            sigreg_loss = proj.new_zeros(())
            for v in range(proj.shape[0]):
                sigreg_loss = sigreg_loss + sigreg(proj[v])
            sigreg_loss = sigreg_loss / proj.shape[0]

            lejepa_loss = (1.0 - cfg.lejepa.lambd) * pred_loss + cfg.lejepa.lambd * sigreg_loss

        metrics_tensors, heavy_stats, proj_hist = jepa_logger.compute_proj_stats(
            proj,
            centers,
            n_global_crops,
            n_local_crops,
            log_heavy,
        )

        if scaler.is_enabled():
            scaler.scale(lejepa_loss).backward()
            scaler.unscale_(optimizer)
            grad_norm_backbone, grad_norm_projector = jepa_logger.module_norms(
                model_root.backbone, model_root.projector, use_grad=True
            )
            if cfg.optim.clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.clip_grad)
            scaler.step(optimizer)
            scaler.update()
            amp_scale = scaler.get_scale()
        else:
            lejepa_loss.backward()
            grad_norm_backbone, grad_norm_projector = jepa_logger.module_norms(
                model_root.backbone, model_root.projector, use_grad=True
            )
            if cfg.optim.clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.clip_grad)
            optimizer.step()
            amp_scale = 1.0
        param_norm_backbone, param_norm_projector = jepa_logger.module_norms(
            model_root.backbone, model_root.projector, use_grad=False
        )
        scheduler.step()

        loss_dict = {
            "pred_loss": pred_loss.detach(),
            "sigreg_loss": sigreg_loss.detach(),
            "lejepa_loss": lejepa_loss.detach(),
        }
        loss_dict_reduced, _ = jepa_logger.log_step(
            iteration=iteration,
            loss_dict=loss_dict,
            metrics_tensors=metrics_tensors,
            heavy_stats=heavy_stats,
            proj_hist=proj_hist,
            optimizer=optimizer,
            lr=lr,
            wd=wd,
            amp_scale=amp_scale,
            grad_norm_backbone=grad_norm_backbone,
            grad_norm_projector=grad_norm_projector,
            param_norm_backbone=param_norm_backbone,
            param_norm_projector=param_norm_projector,
            batch_size=batch_size,
            n_global_crops=n_global_crops,
            n_local_crops=n_local_crops,
            iter_start=iter_start,
            data_time=data_time,
        )
        metric_logger.update(lr=lr)
        metric_logger.update(wd=wd)
        metric_logger.update(**loss_dict_reduced)
        jepa_logger.finish_iter()
        iteration = iteration + 1

    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main():
    if len(sys.argv) != 2:
        raise ValueError("Usage: train_lejepa.py /path/to/config.yaml")
    config_path = sys.argv[1]

    cfg = _load_cfg(config_path)
    cfg = _setup(cfg, config_path)

    model = LeJEPAModel(cfg)
    if cfg.train.use_pretrained:
        _load_pretrained_backbone(cfg, model.backbone)

    device = torch.device("cuda", distributed.get_local_rank())
    torch.cuda.set_device(device)
    model = model.to(device)

    if distributed.get_global_size() > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device],
            output_device=device,
            broadcast_buffers=False,
        )

    sigreg = _build_sigreg(cfg, device)

    _do_train(cfg, model, sigreg)


if __name__ == "__main__":
    main()
