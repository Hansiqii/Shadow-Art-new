import os
from typing import Any, Dict

import torch
import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """从 YAML 文件加载配置为字典。"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(root_dir: str, path_value: str) -> str:
    """把配置里的相对路径转换为绝对路径。"""
    if os.path.isabs(path_value):
        return path_value
    return os.path.join(root_dir, path_value)


def create_optimizer(model: torch.nn.Module, optimizer_cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    """根据配置创建优化器，目前支持 Adam / AdamW。"""
    opt_type = optimizer_cfg.get("type", "adam").lower()
    lr = optimizer_cfg.get("lr", 1e-4)
    weight_decay = optimizer_cfg.get("weight_decay", 0.0)
    betas = optimizer_cfg.get("betas", [0.9, 0.999])
    eps = optimizer_cfg.get("eps", 1e-8)

    if opt_type == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            betas=tuple(betas),
            weight_decay=weight_decay,
            eps=eps,
        )
    elif opt_type == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=tuple(betas),
            weight_decay=weight_decay,
            eps=eps,
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {opt_type}")