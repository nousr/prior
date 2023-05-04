import random
from functools import partial

import torch
import webdataset as wds
from lightning.pytorch.trainer.trainer import Trainer
from torch.utils.data import DataLoader
from torchvision.transforms.functional import center_crop, pil_to_tensor
from open_clip import tokenize

from dalle2_pytorch import DiffusionPrior, DiffusionPriorNetwork, OpenClipAdapter

BATCH_SIZE = 64
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

        text_embedding, text_encoding = prior.clip.embed_text(captions)
        image_embedding, _ = prior.clip.embed_image(images)

        loss = prior.forward(
            image_embed=image_embedding,
            text_embed=text_embedding,
            text_encodings=text_encoding,
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

                text_embedding, text_encoding = prior.clip.embed_text(captions)
                image_embedding, _ = prior.clip.embed_image(images)

                val_loss = prior.forward(
                    image_embed=image_embedding,
                    text_embed=text_embedding,
                    text_encodings=text_encoding,
                )
                print(f"\nValidation Loss: {val_loss.item()}\n")

                predicted_emb = prior.sample(text=captions)

                # embed the images and captions
                image_embeds, _ = prior.clip.embed_image(images)
                text_embeds, _ = prior.clip.embed_text(captions)

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
    prior_xf = DiffusionPriorNetwork(
        dim=512, heads=8, depth=8, normformer=False, rotary_emb=True, max_text_len=77
    ).cuda()

    language_model = OpenClipAdapter(
        name="ViT-B-32", pretrained="laion2b_s34b_b79k"
    ).cuda()

    prior = DiffusionPrior(
        net=prior_xf,
        clip=language_model,
        timesteps=1000,
        image_embed_dim=512,
    ).cuda()

    optimizer = torch.optim.AdamW(
        prior.parameters(), lr=1.1e-4, weight_decay=6e-2, eps=1e-6, betas=(0.9, 0.96)
    )

    print("#--- Loading Dataset ---#")

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

    train(prior, train_dataloader, valid_dataloader, optimizer)


if __name__ == "__main__":
    main()
