# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import sys
from pathlib import Path

from . import vision_transformer as vits


logger = logging.getLogger("dinov2")


def build_model(args, only_teacher=False, img_size=224):
    args.arch = args.arch.removesuffix("_memeff")
    if args.arch == "vit_huge2":
        dinov3_root = Path("./dinov3")
        if not dinov3_root.exists():
            raise AssertionError(f"expected dinov3 repo at {dinov3_root}")
        dinov3_str = str(dinov3_root)
        if dinov3_str not in sys.path:
            sys.path.append(dinov3_str)
        from dinov3.models import vision_transformer as v3_vits

        vit_kwargs = dict(
            img_size=img_size,
            patch_size=args.patch_size,
            in_chans=getattr(args, "in_chans", 3),
            ffn_ratio=getattr(args, "ffn_ratio", 4.0),
            drop_path_rate=getattr(args, "drop_path_rate", 0.0),
            layerscale_init=getattr(args, "layerscale", None),
            norm_layer=getattr(args, "norm_layer", "layernorm"),
            ffn_layer=getattr(args, "ffn_layer", "mlp"),
            ffn_bias=getattr(args, "ffn_bias", True),
            proj_bias=getattr(args, "proj_bias", True),
            n_storage_tokens=getattr(args, "n_storage_tokens", 0),
            mask_k_bias=getattr(args, "mask_k_bias", False),
            pos_embed_rope_base=getattr(args, "pos_embed_rope_base", 100.0),
            pos_embed_rope_min_period=getattr(args, "pos_embed_rope_min_period", None),
            pos_embed_rope_max_period=getattr(args, "pos_embed_rope_max_period", None),
            pos_embed_rope_normalize_coords=getattr(args, "pos_embed_rope_normalize_coords", "separate"),
            pos_embed_rope_shift_coords=getattr(args, "pos_embed_rope_shift_coords", None),
            pos_embed_rope_jitter_coords=getattr(args, "pos_embed_rope_jitter_coords", None),
            pos_embed_rope_rescale_coords=getattr(args, "pos_embed_rope_rescale_coords", None),
            pos_embed_rope_dtype=getattr(args, "pos_embed_rope_dtype", "bf16"),
            untie_cls_and_patch_norms=getattr(args, "untie_cls_and_patch_norms", False),
            untie_global_and_local_cls_norm=getattr(args, "untie_global_and_local_cls_norm", False),
        )
        teacher = v3_vits.__dict__[args.arch](**dict(vit_kwargs, drop_path_rate=0.0))
        if only_teacher:
            return teacher, teacher.embed_dim
        student = v3_vits.__dict__[args.arch](**vit_kwargs)
        embed_dim = student.embed_dim
        return student, teacher, embed_dim
    if "vit" in args.arch:
        vit_kwargs = dict(
            img_size=img_size,
            patch_size=args.patch_size,
            init_values=args.layerscale,
            ffn_layer=args.ffn_layer,
            block_chunks=args.block_chunks,
            qkv_bias=args.qkv_bias,
            proj_bias=args.proj_bias,
            ffn_bias=args.ffn_bias,
            num_register_tokens=args.num_register_tokens,
            interpolate_offset=args.interpolate_offset,
            interpolate_antialias=args.interpolate_antialias,
        )
        teacher = vits.__dict__[args.arch](**vit_kwargs)
        if only_teacher:
            return teacher, teacher.embed_dim
        student = vits.__dict__[args.arch](
            **vit_kwargs,
            drop_path_rate=args.drop_path_rate,
            drop_path_uniform=args.drop_path_uniform,
        )
        embed_dim = student.embed_dim
    return student, teacher, embed_dim


def build_model_from_cfg(cfg, only_teacher=False):
    return build_model(cfg.student, only_teacher=only_teacher, img_size=cfg.crops.global_crops_size)
