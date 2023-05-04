import random
from functools import partial

import torch
import click
import webdataset as wds

from lightning.pytorch.trainer.trainer import Trainer
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf
from open_clip import tokenize
from torch.utils.data import DataLoader
from torchvision.transforms.functional import center_crop, pil_to_tensor

from prior.utils import instantiate_from_config

torch.set_float32_matmul_precision("medium")


def seed_everything(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)


def get_wds_dataset(urls: str, epoch_length: int = None):
    dataset = (
        wds.WebDataset(
            urls=urls, resampled=True, handler=wds.handlers.warn_and_continue
        )
        .shuffle(1000)
        .decode("pil", handler=wds.handlers.warn_and_continue)
        .to_tuple("jpg", "txt")
    )

    if epoch_length is not None:
        dataset = dataset.with_epoch(epoch_length)

    return dataset


def choose_randomly(x):
    return x[torch.randint(len(x), (1,))]


def collate_fn(tokenizer, batch):
    captions = tokenizer([example[1] for example in batch])
    images = [example[0] for example in batch]

    # center crop to 224x224
    images = [
        pil_to_tensor(center_crop(image, (224, 224))).float() / 255 for image in images
    ]

    # stack the images
    images = torch.stack(images)
    images = images.to(memory_format=torch.contiguous_format).float()

    return images, captions


@click.command()
@click.option("--config_path", default="configs/prior_micro.yaml")
@click.option("--seed", default=1337)
@click.option("--devices", default="auto")
@click.option("--num_nodes", default=1)
@click.option("--num_workers", default=32)
@click.option("--fast_dev_run", default=False)
def main(config_path, seed, devices, num_nodes, num_workers, fast_dev_run):
    seed_everything(seed)

    click.secho("#--- Loading Config ---#", fg="green")
    config = OmegaConf.load(config_path)

    click.secho("#--- Creating Model ---#", fg="green")
    prior = instantiate_from_config(config.model)

    click.secho("#--- Loading Dataset ---#", fg="green")
    training_dataset = get_wds_dataset(
        urls="/home/nousr/data/image/laion_coyo_local/{00000..00098}.tar",
        epoch_length=config.trainer.epoch_length,
    )

    validation_dataset = get_wds_dataset(
        urls="/home/nousr/data/image/laion_coyo_local/00099.tar",
        epoch_length=config.trainer.epoch_length,
    )

    # set the tokenizer for the dataloader
    collate = partial(collate_fn, tokenize)

    train_dataloader = DataLoader(
        dataset=training_dataset,
        collate_fn=collate,
        batch_size=config.trainer.batch_size,
        num_workers=num_workers,
    )
    valid_dataloader = DataLoader(
        dataset=validation_dataset,
        collate_fn=collate,
        batch_size=config.trainer.batch_size,
        num_workers=num_workers,
    )

    wandb_logger = WandbLogger(project=config.trainer.project)
    wandb_logger.config = OmegaConf.to_container(config)
    wandb_logger.save(config_path)
    wandb_logger.config.update()

    trainer = Trainer(
        devices=devices,
        num_nodes=num_nodes,
        fast_dev_run=fast_dev_run,
        logger=wandb_logger,
        precision=config.trainer.precision,
        max_epochs=config.trainer.max_epochs,
        gradient_clip_val=config.trainer.gradient_clip_val,
        limit_val_batches=config.trainer.limit_val_batches,
        val_check_interval=config.trainer.val_check_interval,
        accumulate_grad_batches=config.trainer.accumulate_grad_batches,
        enable_checkpointing=config.trainer.enable_checkpointing,
    )

    trainer.fit(
        model=prior,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
    )


if __name__ == "__main__":
    main()
