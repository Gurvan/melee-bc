from typing import Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import os

import torch

from tqdm import tqdm, trange
from accelerate import Accelerator
from ema_pytorch import EMA

from dataset import MeleeDataloader, MeleeInMemoryDataloader
from models import BCModelConfig, BCModel


@dataclass
class TrainConfig:
    overfit_one_batch: bool
    name: str
    log: bool
    datapath: str
    epochs: int
    batch_size: int
    learning_rate: float
    seq_len: int
    checkpoint: str | Path | None
    model_config: BCModelConfig | None
    increase_delay_every_epoch: bool | int

    def __post_init__(self):
        if self.overfit_one_batch:
            self.log = False
        assert (self.model_config is not None) or (self.checkpoint is not None)
        if self.model_config is not None:
            assert self.seq_len > self.model_config.delay


def load_model(config: TrainConfig) -> Tuple[BCModel, BCModelConfig]:
    model_config = config.model_config
    if config.checkpoint is not None:
        params = torch.load(config.checkpoint, map_location="cpu")
        if model_config is None:
            model_config = params["config"]
        model = BCModel(model_config)
        s = {k: v for k, v in params.items() if k != "config"}
        model.load_state_dict(s, strict=True)
    else:
        model = BCModel(model_config)
    return model, model_config


def save_model(name, accelerator, model, model_ema=None):
    path = Path("checkpoints")
    os.makedirs(path, exist_ok=True)
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_with_config(path / name)
    if model_ema is not None:
        unwrapped_model = accelerator.unwrap_model(model_ema.ema_model)
        unwrapped_model.save_with_config(path / (name + "-ema"))


def prepare_data(actions, states, metadata, rewards, epoch=0):
    index_self = epoch % 2
    index_other = (epoch + 1) % 2
    states1 = states[:, index_self]
    states2 = states[:, index_other]
    actions = actions[:, index_self]
    metadata = metadata.int()
    if index_self == 1:
        metadata = metadata[:, [0, 2, 1]]
        rewards = -rewards
    return (
        actions,
        states1,
        states2,
        metadata,
        rewards,
    )


def init_scheduler(config, accelerator, optimizer, steps):
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        epochs=config.epochs,
        steps_per_epoch=steps,
        pct_start=0.01,
    )
    return accelerator.prepare(lr_scheduler)


def train(config: TrainConfig):
    accelerator = Accelerator(
        log_with=["wandb"] if config.log else None, mixed_precision="fp16"
    )
    accelerator.init_trackers(config.name, config=asdict(config))

    model, model_config = load_model(config)
    model_ema = EMA(
        model,
        ema_model=load_model(config)[0],  # init ema to different wieghts
        beta=0.999,
        update_after_step=0,
        update_every=1,
    )

    # if config.checkpoint is None:
    if True:
        loader_type = MeleeDataloader
    else:
        loader_type = MeleeInMemoryDataloader

    data = loader_type(
        seq_len=config.seq_len + 1 + 2 * model.config.delay,
        path=config.datapath,
        data_types=["actions", "states", "metadata"],
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=1e-4
    )

    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        epochs=config.epochs,
        steps_per_epoch=data.len(config.batch_size),
        pct_start=0.01,
    )
    if config.checkpoint is not None:
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1,
            end_factor=1,
            last_epoch=-1,
        )

    model_ema = accelerator.prepare(model_ema)
    model_ema.eval()

    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    if config.overfit_one_batch:
        actions_overfit, states_overfit, metadata_overfit, rewards_overfit = next(
            data.get_samples(
                config.batch_size,
                accelerator.device,
                return_rewards=True,
                low_discrepancy_index=0,
            )
        )
    else:
        actions_overfit, states_overfit, metadata_overfit, rewards_overfit = (
            torch.zeros(1),
            torch.zeros(1),
            torch.zeros(1),
            torch.zeros(1),
        )

    samples_seen = 0
    step = 0
    model.train()
    for epoch in trange(config.epochs):
        progress_bar = tqdm(
            data.get_samples(
                config.batch_size,
                device=accelerator.device,
                return_rewards=True,
                low_discrepancy_index=epoch // 2,
            ),
            total=data.len(config.batch_size),
        )
        for actions, states, metadata, rewards in progress_bar:
            if config.overfit_one_batch:
                actions, states, metadata, rewards = (
                    actions_overfit,
                    states_overfit,
                    metadata_overfit,
                    rewards_overfit,
                )
            optimizer.zero_grad(set_to_none=True)
            with accelerator.autocast():
                (actions, states1, states2, metadata, rewards) = prepare_data(
                    actions,
                    states,
                    metadata,
                    rewards,
                    epoch=epoch,
                )

                loss, aux = model.loss(actions, states1, states2, metadata, rewards)
                loss = loss.mean()

            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if model_ema is not None:
                model_ema.update()
            if not accelerator.optimizer_step_was_skipped:
                lr_scheduler.step()

            if step % 1000 == 0:
                accelerator.wait_for_everyone()
                save_model(
                    config.name + f"-{model.config.delay}",
                    accelerator,
                    model,
                    model_ema,
                )
            samples_seen += actions.shape[0]
            infos = {
                "samples_seen": samples_seen,
                "loss_total": loss,
            }
            s = f"samples seen: {samples_seen} | loss_total: {loss.item():.4f}"
            if config.overfit_one_batch:
                s = "OVERFITTING TEST!!! " + s
            for k, v in aux.items():
                if "logits" in k:
                    continue
                infos[k] = v.mean()
                s += f" | {k}: {infos[k].item():.4f}"
            accelerator.log(infos)
            progress_bar.set_postfix_str(s)
            step += 1
        if config.increase_delay_every_epoch:
            model.config.delay += int(config.increase_delay_every_epoch)
            print()
            print(f"Delay is now {model.config.delay}")
            print(f"EMA Delay is now {model_ema.ema_model.config.delay}")
            print()
            data = MeleeDataloader(
                seq_len=config.seq_len + 1 + 2 * model.config.delay,
                path=config.datapath,
                data_types=["actions", "states", "metadata"],
            )
            lr_scheduler = init_scheduler(
                config, accelerator, optimizer, data.len(config.batch_size)
            )

    accelerator.end_training()
    accelerator.wait_for_everyone()
    save_model(config.name + f"-{model.config.delay}", accelerator, model, model_ema)


if __name__ == "__main__":
    model_config = BCModelConfig(
        memory_dim=512,
        memory_layers=2,
        policy_hidden_dims=1024,
        policy_dim=768,
        delay=1,

        controller_decoder_emb_dim=128,
    )
    config = TrainConfig(
        overfit_one_batch=False,
        # overfit_one_batch=True,
        name="melee-bc-ranked",
        # log=False,
        log=True,
        # datapath="data",
        datapath="/media/DATA/Melee/RankedFizzi/FloatData/",
        epochs=2,
        batch_size=128,
        learning_rate=1e-4,
        seq_len=64,
        # checkpoint=None,  # load existing checkpoint for finetuning
        checkpoint="checkpoints/melee-bc-1-ema",  # load existing checkpoint for finetuning
        model_config=model_config,
        increase_delay_every_epoch=False,
    )

    train(config)
