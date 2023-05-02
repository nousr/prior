import random
from functools import partial

import torch
import webdataset as wds
from lightning.pytorch.trainer.trainer import Trainer
from torch.utils.data import DataLoader
from torchvision.transforms.functional import center_crop, pil_to_tensor
from open_clip import tokenize
from prior import DiffusionPrior, NoiseScheduler, OpenClipAdapter, PriorTransformer

# from dalle2_pytorch import (
#     DiffusionPriorNetwork,
#     DiffusionPrior,
#     OpenClipAdapter,
# )

BATCH_SIZE = 256
CLIP_DIM = 512
CLIP_CONTEXT = 77
SEED = 1337
PARAMETERIZATION = "x0"
SCALE_IMAGE_EMBEDDING = True

torch.set_float32_matmul_precision("medium")
torch.manual_seed(SEED)
random.seed(SEED)


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


def train(prior: DiffusionPrior, train_dataloader, validation_dataloader, optimizer):
    import wandb

    wandb.init(project="prior-testing")

    for step, batch in enumerate(train_dataloader):
        prior.train()
        optimizer.zero_grad()

        images, captions = batch
        images = images.cuda()
        captions = captions.cuda()

        text_embedding, text_encoding = prior.language_model.embed_text(captions)
        image_embedding, _ = prior.language_model.embed_image(images)

        loss = prior.forward(
            text_embedding=text_embedding,
            text_encoding=text_encoding,
            image_embedding=image_embedding,
        )
        loss.backward()
        optimizer.step()

        print(f"Step: {step} | Loss: {loss.item()}")
        wandb.log({"training/loss": loss.item()})

        if step % 250 == 0:
            # sample images
            prior.eval()
            with torch.no_grad():
                sample = next(iter(validation_dataloader))
                images, captions = sample
                images = images.cuda()
                captions = captions.cuda()

                text_embedding, text_encoding = prior.language_model.embed_text(
                    captions
                )
                image_embedding, _ = prior.language_model.embed_image(images)

                val_loss = prior.forward(
                    text_embedding=text_embedding,
                    text_encoding=text_encoding,
                    image_embedding=image_embedding,
                )
                print(f"\nValidation Loss: {val_loss.item()}\n")

                predicted_emb = prior.sample(tokenized_text=captions)

                # embed the images and captions
                image_embeds, _ = prior.language_model.embed_image(images)
                text_embeds, _ = prior.language_model.embed_text(captions)

                # compute the cosine similarity
                text_image_sim = torch.nn.functional.cosine_similarity(
                    image_embeds, text_embeds
                )
                sample_text_sim = torch.nn.functional.cosine_similarity(
                    predicted_emb, text_embeds
                )
                sample_image_sim = torch.nn.functional.cosine_similarity(
                    predicted_emb, image_embeds
                )

                print(f"\nText-Image Similarity: {text_image_sim.mean().item()}")
                print(f"Sample-Text Similarity: {sample_text_sim.mean().item()}")
                print(f"Sample-Image Similarity: {sample_image_sim.mean().item()}\n")

                # log the images
                wandb.log(
                    {
                        "similarity/sample_text_avg": sample_text_sim.mean().item(),
                        "similarity/sample_image_avg": sample_image_sim.mean().item(),
                        "similarity/text_image_avg": text_image_sim.mean().item(),
                        "validation/loss": val_loss.item(),
                    }
                )


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

    # prior_xf = DiffusionPriorNetwork(
    #     dim=512, heads=12, depth=12, normformer=False, rotary_emb=False, max_text_len=77
    # ).cuda()

    scheduler = NoiseScheduler(beta_schedule="cosine", timesteps=1000, loss_type="l2")

    language_model = OpenClipAdapter(
        path="hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
    )
    # language_model = OpenClipAdapter(
    #     name="ViT-B-32",
    #     pretrained="laion2b_s34b_b79k"
    # ).cuda()

    prior = DiffusionPrior(
        prior_transformer=prior_xf,
        noise_scheduler=scheduler,
        language_model=language_model,
        parameterization=PARAMETERIZATION,
        scale_image_embedding=SCALE_IMAGE_EMBEDDING,
    )

    # prior = DiffusionPrior(
    #     net=prior_xf,
    #     clip=language_model,
    #     timesteps=1000,
    #     image_embed_dim=512,
    # ).cuda()

    optimizer = torch.optim.Adam(prior.parameters(), lr=3e-4)

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
    collate = partial(collate_fn, tokenize)

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
        max_epochs=-1,
        precision="bf16-mixed",
        accumulate_grad_batches=1,
        val_check_interval=250,
        limit_val_batches=1,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(
        model=prior,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
    )

    # train(prior, train_dataloader, valid_dataloader, optimizer)


if __name__ == "__main__":
    main()
