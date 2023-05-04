import torch
import wandb
import lightning.pytorch as pl

from tqdm import tqdm
from pprint import pprint

from einops import repeat, rearrange
from torch.nn.functional import cosine_similarity

from prior.prior_transformer import PriorTransformer
from prior.gaussian_diffusion import NoiseScheduler
from prior.adapter import BaseClipAdapter
from prior.utils import instantiate_from_config

def l2norm(t):
    return torch.nn.functional.normalize(t, dim=-1)


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def set_module_requires_grad_(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad


def freeze_all_layers_(module):
    set_module_requires_grad_(module, False)


def freeze_model_and_make_eval_(model):
    model.eval()
    freeze_all_layers_(model)


class DiffusionPrior(pl.LightningModule):
    """
    The DiffusionPrior.

    This model is the bridge between `text` and `image` embeddings.

    Basic Idea:
        - It takes in a `text_embedding`, `text_encoding`, and some `tokens`.

        - Then it will output an `image_embedding` for you to decode.
    """

    def __init__(
        self,
        parameterization,
        scale_embeddings,
        language_model_config,
        noise_scheduler_config,
        prior_transformer_config,
    ):
        super(DiffusionPrior, self).__init__()
        assert parameterization in [
            "eps",
            "x0",
            "v",
        ], f"parameterization must be one of ['eps', 'x0', 'v'] but got {parameterization}"

        self.prior_transformer = instantiate_from_config(prior_transformer_config)
        self.noise_scheduler = instantiate_from_config(noise_scheduler_config)
        self.language_model = instantiate_from_config(language_model_config)
        freeze_model_and_make_eval_(self.language_model)

        self.parameterization = parameterization
        self.scale_embeddings = scale_embeddings

    def p_losses(
        self,
        image_embed: torch.Tensor,
        timesteps: torch.Tensor,
        text_embedding: torch.Tensor,
        text_encoding: torch.Tensor,
    ):
        # get some noise
        noise = torch.rand_like(image_embed)

        # noise the image embedding
        noised_image_embedding = self.noise_scheduler.q_sample(
            x_start=image_embed, t=timesteps, noise=noise
        )

        pred = self.prior_transformer.forward(
            x=noised_image_embedding,
            timesteps=timesteps,
            text_emb=text_embedding,
            text_enc=text_encoding,
        )

        # calculate the loss, depending on the parameterization
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = image_embed
        elif self.parameterization == "v":
            target = self.noise_scheduler.calculate_v(
                x_start=image_embed, t=timesteps, noise=noise
            )
        else:
            raise ValueError(
                f"parameterization must be one of ['eps', 'x0', 'v'] but got {self.parameterization}"
            )

        return self.noise_scheduler.loss_fn(pred, target)

    def forward(
        self,
        text_embedding: torch.Tensor,
        text_encoding: torch.Tensor,
        image_embedding: torch.Tensor,
    ):
        """
        Predict the `image_embedding` given the following:
            - `text_embedding`: text embedding from your LLM.
            - `text_encoding`: text encoding from your LLM.

        Returns:
            - `loss`: the calculated loss of your model.
            - `image_embedding`: the predicted image embedding.
        """
        batch_size = image_embedding.shape[0]

        # get the timesteps to train on
        timesteps = self.noise_scheduler.sample_random_times(batch_size)

        # scale the image embedding
        if self.scale_embeddings:
            image_embedding = l2norm(image_embedding) * self.language_model.dim_latent**0.5
            text_embedding = l2norm(text_embedding) * self.language_model.dim_latent**0.5
            text_encoding = l2norm(text_encoding) * self.language_model.dim_latent**0.5

        # send to p_losses & return loss
        return self.p_losses(
            image_embedding,
            timesteps,
            text_embedding=text_embedding,
            text_encoding=text_encoding,
        )

    def setup(self, stage: str):
        # initialize wandb on rank 0
        if stage == "fit" and self.trainer.is_global_zero:
            wandb.init(project="prior-testing")

    def training_step(self, batch, _):
        # get the text embedding and encoding
        image, tokenized_caption = batch
        text_embedding, text_encoding = self.language_model.embed_text(
            tokenized_caption
        )
        # get the image embedding
        # TODO: try conditioning on the image encoding as well
        image_embedding, _ = self.language_model.embed_image(image)

        loss = self.forward(
            text_embedding=text_embedding,
            text_encoding=text_encoding,
            image_embedding=image_embedding,
        )

        if self.trainer.is_global_zero:
            wandb.log({"training/loss": loss})

        # forward pass
        return loss

    @torch.no_grad()
    def find_cosine_similarity(self, emb_0, emb_1):
        """
        Find the cosine similarity between the two embeddings.
        """

        # normalize the embeddings
        emb_0 = emb_0 / emb_0.norm(dim=-1, keepdim=True)
        emb_1 = emb_1 / emb_1.norm(dim=-1, keepdim=True)

        # calculate the cosine similarity
        return cosine_similarity(emb_0, emb_1, dim=-1)

    @torch.no_grad()
    def p_mean_variance(self, x, t, text_embedding, text_encoding, cond_scale):
        # TODO: check that model was trained with dropout
        # TODO: do classifier free guidance
        predicted_tokens = self.prior_transformer.forward(
            x, t, text_embedding, text_encoding
        )

        if self.parameterization == "v":
            x_start = self.noise_scheduler.predict_start_from_v(
                x, t=t, v=predicted_tokens
            )
        elif self.parameterization == "x0":
            x_start = predicted_tokens
        elif self.parameterization == "eps":
            x_start = self.noise_scheduler.predict_start_from_noise(
                x_t=x, t=t, noise=predicted_tokens
            )
        else:
            raise ValueError(
                f"parameterization must be one of ['eps', 'x0', 'v'] but got {self.parameterization}"
            )

        model_mean, _, posterior_log_variance = self.noise_scheduler.q_posterior(
            x_start=x_start, x_t=x, t=t
        )

        return model_mean, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, text_embedding, text_encoding, cond_scale):
        model_mean, model_log_variance = self.p_mean_variance(
            x, t, text_embedding, text_encoding, cond_scale
        )

        noise = torch.randn_like(x)

        # only noise if t > 0
        nonzero_mask = (1 - (t == 0).float()).reshape(
            x.shape[0], *((1,) * (len(x.shape) - 1))
        )
        pred_x = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

        return pred_x

    @torch.no_grad()
    def p_sample_loop_ddpm(self, text_embedding, text_encoding, cond_scale, steps):
        """
        Sample from the prior using DDPM.
        """
        batch_size = text_embedding.shape[0]
        embed_dim = text_embedding.shape[-1]
        device = text_embedding.device

        # initialize the image embedding
        image_embed = torch.randn(size=(batch_size, embed_dim)).to(device)

        for step in tqdm(range(steps)[::-1], desc="ddpm sampling loop", total=steps):
            times = torch.full(
                size=(batch_size,), fill_value=step, device=device, dtype=torch.long
            )
            image_embed = self.p_sample(
                x=image_embed,
                t=times,
                text_embedding=text_embedding,
                text_encoding=text_encoding,
                cond_scale=cond_scale,
            )

        return image_embed

    @torch.no_grad()
    def p_sample_loop(self, text_embedding, text_encoding, cond_scale, steps):
        assert (
            steps <= self.noise_scheduler.num_timesteps
        ), f"timesteps must be <= {self.noise_scheduler.timesteps} but got {steps}"

        if steps < self.noise_scheduler.num_timesteps:
            raise NotImplementedError("ddim sampling is not yet implemented")
        else:
            image_embedding = self.p_sample_loop_ddpm(
                text_embedding=text_embedding,
                text_encoding=text_encoding,
                cond_scale=cond_scale,
                steps=steps,
            )

        if self.scale_embeddings:
            image_embedding /= self.language_model.dim_latent**0.5

        return image_embedding

    @torch.no_grad()
    def sample(
        self,
        tokenized_text: torch.Tensor,
        steps: int = None,
        cond_scale: float = 1.0,
        best_of: int = 2,
    ):
        """
        Sample N image embeddings for each caption.
        Return the best one.
        """
        steps = default(steps, self.noise_scheduler.num_timesteps)

        # repeat the tokenized text N times
        # tokenized_text = repeat(tokenized_text, "b ... -> (b r) ...", r=best_of)

        # embed the text
        text_embedding, text_encoding = self.language_model.embed_text(tokenized_text)

        if self.scale_embeddings:
            text_embedding = l2norm(text_embedding) * self.language_model.dim_latent**0.5
            text_encoding = l2norm(text_encoding) * self.language_model.dim_latent**0.5

        # predict the image embedding
        image_embedding = self.p_sample_loop(
            text_embedding=text_embedding,
            text_encoding=text_encoding,
            cond_scale=cond_scale,
            steps=steps,
        )
        # # reshape the embeddings to be (batch_size, best_of, ...)
        # text_embedding = rearrange(text_embedding, "(b r) ... -> b r ...", r=best_of)
        # image_embedding = rearrange(image_embedding, "(b r) ... -> b r ...", r=best_of)

        # # find the cosine similarity with the caption
        # cosine_similarity = torch.einsum(
        #     "b r d, b r d -> b r", l2norm(text_embedding), l2norm(image_embedding)
        # )

        # # find the best image embedding for each caption
        # top_indices = cosine_similarity.topk(k=1).indices
        # top_indices = repeat(top_indices, "b 1 -> b 1 d", d=image_embedding.shape[-1])

        # # gather the best image embeddings
        # best_image_embeddings = image_embedding.gather(dim=1, index=top_indices)

        return image_embedding

    @torch.no_grad()
    def validation_step(self, batch, _):
        # get the text embedding and encoding
        image, tokenized_caption = batch

        # only sample 32 validation prompts
        image = image[:32, ...]
        tokenized_caption = tokenized_caption[:32, ...]

        text_embedding, text_encoding = self.language_model.embed_text(
            tokenized_caption
        )

        # simulate an unrelated text embedding by rolling the text embedding
        unrelated_text_embedding = torch.roll(text_embedding, 1, dims=0)

        # get the image embedding
        image_embedding, _ = self.language_model.embed_image(image)

        loss = self.forward(
            text_embedding=text_embedding,
            text_encoding=text_encoding,
            image_embedding=image_embedding,
        )

        # ------------------------------
        # now actually sample embeddings
        # ------------------------------
        predicted_image_embeddings = self.sample(
            tokenized_text=tokenized_caption,
        )
        # ------------------------------

        # ------------------------------------------------------------------
        # compute the average cosine similarity between:
        # ------------------------------------------------------------------
        #   - the text embedding and the original image embedding
        #   - the text embedding and the sampled image embedding
        #   - the sampled image embedding and the original image embedding
        # ------------------------------------------------------------------
        cosine_sim_report = {}
        for name, emb_0, emb_1 in [
            ("similarity/text_image", text_embedding, image_embedding),
            ("similarity/sample_text", text_embedding, predicted_image_embeddings),
            (
                "similarity/sample_unrelated_text",
                unrelated_text_embedding,
                predicted_image_embeddings,
            ),
            ("similarity/sample_image", predicted_image_embeddings, image_embedding),
        ]:
            cosine_sim = self.find_cosine_similarity(emb_0=emb_0, emb_1=emb_1)
            cosine_sim_report[f"{name}_avg"] = cosine_sim.mean().item()
        # ------------------------------------------------------------------

        if self.trainer.is_global_zero:
            wandb.log({"validation/loss": loss, **cosine_sim_report})
            print()  # newline for readability
            pprint(cosine_sim_report)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        return optimizer
