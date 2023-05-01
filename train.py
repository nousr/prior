import random
from functools import partial

import datasets
import torch
import webdataset as wds
from lightning.pytorch.trainer.trainer import Trainer
from torch.utils.data import DataLoader
from torchvision.transforms.functional import center_crop, pil_to_tensor

from prior import DiffusionPrior, NoiseScheduler, OpenClipAdapter, PriorTransformer

BATCH_SIZE = 128
CLIP_DIM = 512
CLIP_CONTEXT = 77
SEED = 1337
PARAMETERIZATION = "x0"
SCALE_IMAGE_EMBEDDING = True

datasets.logging.set_verbosity_error()
torch.set_float32_matmul_precision("medium")
torch.manual_seed(SEED)
random.seed(SEED)


def choose_randomly(x):
    return x[torch.randint(len(x), (1,))]


def collate_fn(tokenizer, batch):
    captions = tokenizer([example[1] for example in batch])
    images = [example[0] for example in batch]

    # center crop to 224x224
    images = [pil_to_tensor(center_crop(image, (224, 224))).float() / 255 for image in images]

    # stack the images
    images = torch.stack(images)
    images = images.to(memory_format=torch.contiguous_format).float()

    return images, captions


def main():
    print("#--- Creating Model ---#")
    prior_xf = PriorTransformer(
        ctx_len=CLIP_CONTEXT,
        emb_dim=768,
        num_layers=12,
        num_heads=12,
        final_ln=True,
        clip_dim=CLIP_DIM,
    )

    scheduler = NoiseScheduler(beta_schedule="cosine", timesteps=1000, loss_type="l2")

    language_model = OpenClipAdapter(
        path="hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
    )

    prior = DiffusionPrior(
        prior_transformer=prior_xf,
        noise_scheduler=scheduler,
        language_model=language_model,
        parameterization=PARAMETERIZATION,
        scale_image_embedding=SCALE_IMAGE_EMBEDDING,
    )

    print("#--- Loading Dataset ---#")
    # dataset = datasets.load_dataset("fusing/wikiart_captions")
    training_dataset = (
        wds.WebDataset(
            urls="/home/nousr/data/image/laion_coyo_local/{00000..00048}.tar"
        )
        .shuffle(1000)
        .decode("pil", handler=wds.handlers.warn_and_continue)
        .to_tuple("jpg", "txt")
    )
    validation_dataset = (
        wds.WebDataset(urls="/home/nousr/data/image/laion_coyo_local/00049.tar")
        .shuffle(1000)
        .decode("pil", handler=wds.handlers.warn_and_continue)
        .to_tuple("jpg", "txt")
    )

    # set the tokenizer for the dataloader
    collate = partial(collate_fn, language_model.tokenize_text)

    train_dataloader = DataLoader(
        dataset=training_dataset,
        collate_fn=collate,
        batch_size=BATCH_SIZE,
        num_workers=32,
        pin_memory=False,
    )
    valid_dataloader = DataLoader(
        dataset=validation_dataset,
        collate_fn=collate,
        batch_size=BATCH_SIZE,
        num_workers=32,
        pin_memory=False,
    )

    trainer = Trainer(
        max_steps=-1,
        precision="bf16-mixed",
        accumulate_grad_batches=1,
        val_check_interval=1000,
        limit_val_batches=1,
        logger=False,
        enable_checkpointing=False
    )
    trainer.fit(
        model=prior,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
    )


if __name__ == "__main__":
    main()
