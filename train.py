import random
import pkg_resources
from functools import partial
from datetime import timedelta

import click
import torch
import webdataset as wds

from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.trainer.trainer import Trainer
from omegaconf import OmegaConf
from open_clip import tokenize
from torch.utils.data import DataLoader
from torchvision import transforms

from prior.utils import instantiate_from_config
import kornia


torch.set_float32_matmul_precision("medium")

# sd - unclip compatible augmentation pipeline
assert (
    pkg_resources.get_distribution("kornia").version == "0.6.8"
), f"SD-Unclip requires kornia==0.6.8"
AUGMENTATION_PIPELINE = transforms.Compose(
    [
        transforms.ToTensor(),  # pil->tensor (0, 1)
        transforms.Lambda(lambda x: x * 2.0 - 1.0),  # (0, 1) -> (-1, 1)
        transforms.Lambda(  # resize to 224x224
            lambda x: kornia.geometry.resize(
                x,
                (224, 224),
                interpolation="bicubic",
                align_corners=True,
                antialias=True,
            )
        ),
        transforms.Lambda(lambda x: (x + 1.0) / 2.0),  # (-1, 1) -> (0, 1)
        transforms.Lambda(  # normalize according to CLIP
            lambda x: kornia.enhance.normalize(
                data=x.unsqueeze(0),
                mean=torch.tensor([0.48145466, 0.4578275, 0.40821073]),
                std=torch.tensor([0.26862954, 0.26130258, 0.27577711]),
            )
        ),
    ]
)


def seed_everything(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)


def get_wds_dataset(urls: str, epoch_length: int = None):
    def filter_missing(x):
        return ("json" in x and "jpg" in x) and ("caption" in x["json"]) and (x["json"]["caption"] != None)

    dataset = (
        wds.WebDataset(
            urls=urls, resampled=True, handler=wds.handlers.warn_and_continue
        )
        .shuffle(1000)
        .decode("pil", handler=wds.handlers.warn_and_continue)
        .select(filter_missing)
        .to_tuple("jpg", "json")
        .map_tuple(AUGMENTATION_PIPELINE, lambda x: tokenize(x["caption"]))
    )

    if epoch_length is not None:
        dataset = dataset.with_epoch(epoch_length)

    return dataset


def get_dataloader(dataset, batch_size, num_workers, collate_fn):
    return DataLoader(
        dataset=dataset,
        collate_fn=collate_fn,
        batch_size=batch_size,
        num_workers=num_workers,
    )


def collate_fn(batch):
    images = (
        torch.concat([x[0] for x in batch])
        .to(memory_format=torch.contiguous_format)
        .float()
    )
    captions = torch.concat([x[1] for x in batch])

    return images, captions


@click.command()
@click.option("--config_path", default="configs/prior_micro.yaml")
@click.option("--seed", default=1337)
@click.option("--devices", default="auto")
@click.option("--num_nodes", default=1)
@click.option("--num_workers", default=6)
@click.option("--fast_dev_run", is_flag=True, default=False)
def main(config_path, seed, devices, num_nodes, num_workers, fast_dev_run):
    seed_everything(seed)

    config = OmegaConf.load(config_path)

    prior = instantiate_from_config(config.model, config_path=config_path)

    training_dataset = get_wds_dataset(
        urls=config.trainer.train_data_urls,
        epoch_length=config.trainer.epoch_length,
    )

    validation_dataset = get_wds_dataset(
        urls=config.trainer.val_data_urls,
        epoch_length=config.trainer.epoch_length,
    )

    train_dataloader = get_dataloader(
        training_dataset, config.trainer.train_batch_size, num_workers, collate_fn
    )

    valid_dataloader = get_dataloader(
        validation_dataset, config.trainer.valid_batch_size, num_workers, collate_fn
    )

    # --- Create Trainer --- #

    callbacks = [LearningRateMonitor(logging_interval="step")]

    if config.trainer.enable_checkpointing:
        callbacks.append(
            ModelCheckpoint(
                dirpath=config.trainer.checkpoint_dirpath,
                save_top_k=config.trainer.checkpoint_save_top_k,
                monitor=config.trainer.checkpoint_monitor,
                mode=config.trainer.checkpoint_mode,
                filename=config.trainer.checkpoint_filename,
                save_last=True,
                train_time_interval=timedelta(
                    minutes=config.trainer.checkpoint_train_time_interval_minutes
                ),
                save_on_train_epoch_end=True,
            )
        )

    trainer = Trainer(
        devices=devices,
        num_nodes=num_nodes,
        fast_dev_run=fast_dev_run,
        logger=True,
        precision=config.trainer.precision,
        max_epochs=config.trainer.max_epochs,
        gradient_clip_val=config.trainer.gradient_clip_val,
        limit_val_batches=config.trainer.limit_val_batches,
        val_check_interval=config.trainer.val_check_interval,
        accumulate_grad_batches=config.trainer.accumulate_grad_batches,
        enable_checkpointing=config.trainer.enable_checkpointing,
        callbacks=callbacks,
    )

    trainer.fit(
        model=prior,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
        ckpt_path=config.trainer.ckpt_path,
    )


if __name__ == "__main__":
    main()
