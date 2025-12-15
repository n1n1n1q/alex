import os
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor

# ...existing code...
from minestudio.data import EventDataModule
from minestudio.data.minecraft.callbacks import (
    ImageKernelCallback,
    ActionKernelCallback,
)
from minestudio.offline import MineLightning
from minestudio.models import load_vpt_policy, VPTPolicy
from minestudio.offline.mine_callbacks import BehaviorCloneCallback
from minestudio.offline.lightning_callbacks import (
    SmartCheckpointCallback,
    SpeedMonitorCallback,
)
from pathlib import Path
from minestudio.data.minecraft.utils import pull_datasets_from_remote

import argparse

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

policy = VPTPolicy.from_pretrained("CraftJarvis/MineStudio_VPT.foundation_model_3x")
mine_lightning = MineLightning(
    mine_policy=policy,
    learning_rate=0.00004,
    warmup_steps=2000,
    weight_decay=0.000181,
    callbacks=[BehaviorCloneCallback(weight=0.01)],
)


def _resolve_dataset_dirs():
    """
    Return a list of valid dataset dirs.
    Prefer a local checkout if it contains a non-empty LMDB; otherwise download '10xx'.
    """
    local_dir = Path(__file__).resolve().parents[2] / "minestudio-data-10xx-v110"
    event_mdb = local_dir / "event" / "data.mdb"
    try:
        if event_mdb.exists() and event_mdb.stat().st_size > 1_000_000:
            return [str(local_dir)]
    except Exception:
        pass
    return pull_datasets_from_remote(["10xx"])


DATASET_DIRS = _resolve_dataset_dirs()

mine_data = EventDataModule(
    data_params=dict(
        dataset_dirs=DATASET_DIRS,
        event_paths=[str(Path(p) / "event") for p in DATASET_DIRS],
        modal_kernel_callbacks=[
            ImageKernelCallback(
                frame_width=128,
                frame_height=128,
            ),
            ActionKernelCallback(),
        ],
        win_len=32,
        split_ratio=0.9,
        event_regex="minecraft.kill_entity:.*",
        bias=16,
        min_nearby=32,
    ),
    batch_size=1,
    num_workers=0,
    prefetch_factor=None,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--path_ckpt", type=str, default=None, help="Path to checkpoint to resume from"
)

ckpt_path = parser.parse_args().path_ckpt
L.Trainer(
    logger=WandbLogger(project="minestudio-vpt"),
    devices=1,
    precision="bf16-mixed",
    gradient_clip_val=1.0,
    accumulate_grad_batches=8,
    callbacks=[
        LearningRateMonitor(logging_interval="step"),
        SpeedMonitorCallback(),
        SmartCheckpointCallback(
            dirpath="./weights",
            filename="weight-{epoch}-{step}",
            save_top_k=-1,
            every_n_train_steps=2000,
            save_weights_only=True,
        ),
        SmartCheckpointCallback(
            dirpath="./checkpoints",
            filename="ckpt-{epoch}-{step}",
            save_top_k=1,
            every_n_train_steps=2000 + 1,
            save_weights_only=False,
        ),
    ],
).fit(model=mine_lightning, datamodule=mine_data, ckpt_path=ckpt_path, max_epochs=20)
