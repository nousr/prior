import torch
import wandb
import lightning.pytorch as pl

from tqdm import tqdm
from pprint import pprint
from contextlib import contextmanager

from einops import repeat, rearrange
from torch.nn.functional import cosine_similarity

from prior.ema import LitEma
from prior.utils import (
    instantiate_from_config,
    get_obj_from_str,
    eval_decorator,
    load_stats,
)


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
        optimizer_config,
        lr_scheduler_config,
        language_model_config,
        noise_scheduler_config,
        prior_transformer_config,
        image_embedding_stats_path,
        config_path=None,
        use_ema=True,
    ):
        super(DiffusionPrior, self).__init__()
        self.config_path = config_path

        assert parameterization in [
            "eps",
            "x0",
            "v",
        ], f"parameterization must be one of ['eps', 'x0', 'v'] but got {parameterization}"

        self.prior_transformer = instantiate_from_config(prior_transformer_config)
        self.noise_scheduler = instantiate_from_config(noise_scheduler_config)
        self.language_model = instantiate_from_config(language_model_config)
        freeze_model_and_make_eval_(self.language_model)

        self.use_ema = use_ema

        if self.use_ema:
            self.ema_model = LitEma(self.prior_transformer)

        self.optimizer_config = optimizer_config
        self.lr_scheduler_config = lr_scheduler_config

        self.parameterization = parameterization

        # load the stats
        self.image_embedding_stats_path = image_embedding_stats_path
        mu, std = load_stats(self.image_embedding_stats_path)
        self.register_buffer("image_embedding_mu", mu.unsqueeze(0), persistent=True)
        self.register_buffer("image_embedding_std", std.unsqueeze(0), persistent=True)

    def scale_image_embedding(self, image_embedding):
        return (image_embedding - self.image_embedding_mu) / self.image_embedding_std

    def unscale_image_embedding(self, image_embedding):
        return (image_embedding * self.image_embedding_std) + self.image_embedding_mu

    def scale_text_embedding(self, text_embedding):
        raise NotImplementedError

    def unscale_text_embedding(self, text_embedding):
        raise NotImplementedError

    def setup(self, stage: str):
        # initialize wandb on rank 0
        if stage == "fit" and self.trainer.is_global_zero:
            wandb.init(project="prior-testing")
            if exists(self.config_path):
                wandb.save(self.config_path)

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
        image_embedding = self.scale_image_embedding(image_embedding)

        # send to p_losses & return loss
        return self.p_losses(
            image_embedding,
            timesteps,
            text_embedding=text_embedding,
            text_encoding=text_encoding,
        )

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
            wandb.log(
                {"training/loss": loss, "trainer/global_step": self.global_step},
                step=self.global_step,
            )

        # forward pass
        return loss

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.ema_model(self.prior_transformer)

        # log the learning rate
        if self.trainer.is_global_zero:
            wandb.log(
                {"training/lr": self.optimizers().param_groups[0]["lr"]},
                step=self.global_step,
            )

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.ema_model.store(self.prior_transformer.parameters())
            self.ema_model.copy_to(self.prior_transformer)
            if context is not None:
                print(f"\n{context}: Switched to EMA weights\n")
        try:
            yield None
        finally:
            if self.use_ema:
                self.ema_model.restore(self.prior_transformer.parameters())
                if context is not None:
                    print(f"\n{context}: Restored training weights\n")

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

        image_embedding = self.unscale_image_embedding(image_embedding)

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
        tokenized_text = repeat(tokenized_text, "b ... -> (b r) ...", r=best_of)

        # embed the text
        text_embeds, text_encodings = self.language_model.embed_text(tokenized_text)

        # if self.scale_embeddings:
        # text_embeds = l2norm(text_embeds) * self.language_model.dim_latent**0.5
        # text_encoding = (
        #     l2norm(text_encodings) * self.language_model.dim_latent**0.5
        # )

        # predict the image embedding
        image_embeds = self.p_sample_loop(
            text_embedding=text_embeds,
            text_encoding=text_encodings,
            cond_scale=cond_scale,
            steps=steps,
        )

        text_embeds = rearrange(text_embeds, "(b r) d -> b r d", r=best_of)
        image_embeds = rearrange(image_embeds, "(b r) d -> b r d", r=best_of)

        text_image_sims = torch.torch.einsum(
            "b r d, b r d -> b r", l2norm(text_embeds), l2norm(image_embeds)
        )
        top_sim_indices = text_image_sims.topk(k=1).indices

        top_sim_indices = repeat(
            top_sim_indices, "b 1 -> b 1 d", d=self.language_model.dim_latent
        )

        top_image_embeds = image_embeds.gather(1, top_sim_indices)

        return rearrange(top_image_embeds, "b 1 d -> b d")

    @torch.no_grad()
    def validation_step(self, batch, _):
        # get the text embedding and encoding
        image, tokenized_caption = batch

        text_embedding, text_encoding = self.language_model.embed_text(
            tokenized_caption
        )

        # simulate an unrelated text embedding by rolling the text embedding
        unrelated_text_embedding = torch.roll(text_embedding, 1, dims=0)

        # get the image embedding
        image = (image + 1.0) / 2.0
        image_embedding, _ = self.language_model.embed_image(image)

        with self.ema_scope("Validation Step"):
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
            cosine_sim = torch.nn.functional.cosine_similarity(emb_0, emb_1)
            cosine_sim_report[f"{name}_avg"] = cosine_sim.mean().item()
        # ------------------------------------------------------------------

        if self.trainer.is_global_zero:
            wandb.log({"validation/loss": loss, **cosine_sim_report})
            print()  # newline for readability
            pprint(cosine_sim_report)

    def configure_optimizers(self):
        optimizer = get_obj_from_str(self.optimizer_config.target)(
            self.parameters(), **self.optimizer_config.get("params", dict())
        )
        scheduler = get_obj_from_str(self.lr_scheduler_config.target)(
            **self.lr_scheduler_config.get("params", dict())
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=scheduler.schedule
        )

        schedulers = [{"scheduler": scheduler, "interval": "step", "frequency": 1}]

        return [optimizer], schedulers


class LegacyDiffusionPrior(pl.LightningModule):
    def __init__(
        self,
        prior_transformer_config,
        language_model_config,
        noise_scheduler_config,
        optimizer_config,
        lr_scheduler_config,
        image_embedding_stats_path,
        config_path=None,
        use_ema=True,
        image_channels=3,
        sample_timesteps=None,
        cond_drop_prob=0.0,
        text_cond_drop_prob=None,
        image_cond_drop_prob=None,
        predict_x_start=True,
        predict_v=False,
        condition_on_text_encodings=True,  # the paper suggests this is needed, but you can turn it off for your CLIP preprocessed text embed -> image embed training
        sampling_clamp_l2norm=False,  # whether to l2norm clamp the image embed at each denoising iteration (analogous to -1 to 1 clipping for usual DDPMs)
        sampling_final_clamp_l2norm=False,  # whether to l2norm the final image embedding output (this is also done for images in ddpm)
        training_clamp_l2norm=False,
        init_image_embed_l2norm=False,
    ):
        super().__init__()
        self.config_path = config_path
        self.sample_timesteps = sample_timesteps

        self.net = instantiate_from_config(prior_transformer_config)
        self.noise_scheduler = instantiate_from_config(noise_scheduler_config)
        self.language_model = instantiate_from_config(language_model_config)
        freeze_model_and_make_eval_(self.language_model)

        self.use_ema = use_ema

        if self.use_ema:
            self.ema_model = LitEma(self.net)

        self.optimizer_config = optimizer_config
        self.lr_scheduler_config = lr_scheduler_config

        self.image_embed_dim = self.net.clip_dim

        assert (
            not exists(self.language_model)
            or self.language_model.dim_latent == self.image_embed_dim
        ), f"you passed in a CLIP to the diffusion prior with latent dimensions of {self.language_model.dim_latent}, but your image embedding dimension (keyword image_embed_dim) for the DiffusionPrior was set to {self.image_embed_dim}"

        self.channels = default(
            image_channels, lambda: self.language_model.image_channels
        )
        self.text_cond_drop_prob = default(text_cond_drop_prob, cond_drop_prob)
        self.image_cond_drop_prob = default(image_cond_drop_prob, cond_drop_prob)

        self.can_classifier_guidance = (
            self.text_cond_drop_prob > 0.0 and self.image_cond_drop_prob > 0.0
        )
        self.condition_on_text_encodings = condition_on_text_encodings

        # in paper, they do not predict the noise, but predict x0 directly for image embedding, claiming empirically better results. I'll just offer both.

        self.predict_x_start = predict_x_start
        self.predict_v = predict_v  # takes precedence over predict_x_start

        # load the stats
        self.image_embedding_stats_path = image_embedding_stats_path
        mu, std = load_stats(self.image_embedding_stats_path)
        self.register_buffer("image_embedding_mu", mu.unsqueeze(0), persistent=True)
        self.register_buffer("image_embedding_std", std.unsqueeze(0), persistent=True)

        # whether to force an l2norm, similar to clipping denoised, when sampling

        self.sampling_clamp_l2norm = sampling_clamp_l2norm
        self.sampling_final_clamp_l2norm = sampling_final_clamp_l2norm

        self.training_clamp_l2norm = training_clamp_l2norm
        self.init_image_embed_l2norm = init_image_embed_l2norm

        # device tracker

        self.register_buffer("_dummy", torch.tensor([True]), persistent=False)

    def scale_image_embedding(self, image_embedding):
        return (image_embedding - self.image_embedding_mu) / self.image_embedding_std

    def unscale_image_embedding(self, image_embedding):
        return (image_embedding * self.image_embedding_std) + self.image_embedding_mu

    def scale_text_embedding(self, text_embedding):
        raise NotImplementedError

    def unscale_text_embedding(self, text_embedding):
        raise NotImplementedError

    def setup(self, stage: str):
        # initialize wandb on rank 0
        if stage == "fit" and self.trainer.is_global_zero:
            wandb.init(project="prior-testing")
            if exists(self.config_path):
                wandb.save(self.config_path)

    def training_step(self, batch, _):
        # get the text embedding and encoding
        image, tokenized_caption = batch
        text_embedding, text_encoding = self.language_model.embed_text(
            tokenized_caption
        )

        image_embedding, _ = self.language_model.embed_image(image)

        loss = self.forward(
            text_embed=text_embedding,
            text_encodings=text_encoding,
            image_embed=image_embedding,
        )

        if self.trainer.is_global_zero:
            wandb.log(
                {"training/loss": loss, "trainer/global_step": self.global_step},
                step=self.global_step,
            )

        # forward pass
        return loss

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.ema_model.store(self.net.parameters())
            self.ema_model.copy_to(self.net)
            if context is not None:
                print(f"\n{context}: Switched to EMA weights\n")
        try:
            yield None
        finally:
            if self.use_ema:
                self.ema_model.restore(self.net.parameters())
                if context is not None:
                    print(f"\n{context}: Restored training weights\n")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.ema_model(self.net)

        # log the learning rate
        if self.trainer.is_global_zero:
            wandb.log(
                {"training/lr": self.optimizers().param_groups[0]["lr"]},
                step=self.global_step,
            )

    @torch.no_grad()
    def validation_step(self, batch, _):
        # get the text embedding and encoding
        image, tokenized_caption = batch

        text_embedding, text_encoding = self.language_model.embed_text(
            tokenized_caption
        )

        # simulate an unrelated text embedding by rolling the text embedding
        unrelated_text_embedding = torch.roll(text_embedding, 1, dims=0)

        # get the image embedding
        image = (image + 1.0) / 2.0
        image_embedding, _ = self.language_model.embed_image(image)

        with self.ema_scope("Validation Step"):
            loss = self.forward(
                text_embed=text_embedding,
                text_encodings=text_encoding,
                image_embed=image_embedding,
            )

            # ------------------------------
            # now actually sample embeddings
            # ------------------------------
            predicted_image_embeddings = self.sample(
                text=tokenized_caption,
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
            cosine_sim = torch.nn.functional.cosine_similarity(emb_0, emb_1)
            cosine_sim_report[f"{name}_avg"] = cosine_sim.mean().item()
        # ------------------------------------------------------------------

        if self.trainer.is_global_zero:
            wandb.log({"validation/loss": loss, **cosine_sim_report})
            print()  # newline for readability
            pprint(cosine_sim_report)

    def configure_optimizers(self):
        optimizer = get_obj_from_str(self.optimizer_config.target)(
            self.parameters(), **self.optimizer_config.get("params", dict())
        )
        scheduler = get_obj_from_str(self.lr_scheduler_config.target)(
            **self.lr_scheduler_config.get("params", dict())
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=scheduler.schedule
        )

        schedulers = [{"scheduler": scheduler, "interval": "step", "frequency": 1}]

        return [optimizer], schedulers

    @property
    def device(self):
        return self._dummy.device

    def l2norm_clamp_embed(self, image_embed):
        return l2norm(image_embed) * self.image_embed_scale

    def p_mean_variance(
        self, x, t, text_cond, self_cond=None, clip_denoised=False, cond_scale=1.0
    ):
        assert not (
            cond_scale != 1.0 and not self.can_classifier_guidance
        ), "the model was not trained with conditional dropout, and thus one cannot use classifier free guidance (cond_scale anything other than 1)"

        pred = self.net.forward(
            x,
            t,
            text_emb=text_cond["text_embed"],
            text_enc=text_cond["text_encodings"],
        )

        if self.predict_v:
            x_start = self.noise_scheduler.predict_start_from_v(x, t=t, v=pred)
        elif self.predict_x_start:
            x_start = pred
        else:
            x_start = self.noise_scheduler.predict_start_from_noise(x, t=t, noise=pred)

        if clip_denoised and not self.predict_x_start:
            x_start.clamp_(-1.0, 1.0)

        if self.predict_x_start and self.sampling_clamp_l2norm:
            x_start = l2norm(x_start) * self.image_embed_scale

        (
            model_mean,
            posterior_variance,
            posterior_log_variance,
        ) = self.noise_scheduler.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(
        self, x, t, text_cond=None, self_cond=None, clip_denoised=True, cond_scale=1.0
    ):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x,
            t=t,
            text_cond=text_cond,
            self_cond=self_cond,
            clip_denoised=clip_denoised,
            cond_scale=cond_scale,
        )
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        pred = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred, x_start

    @torch.no_grad()
    def p_sample_loop_ddpm(self, shape, text_cond, cond_scale=1.0):
        batch, device = shape[0], self.device

        image_embed = torch.randn(shape, device=device)
        x_start = None  # for self-conditioning

        if self.init_image_embed_l2norm:
            image_embed = l2norm(image_embed) * self.image_embed_scale

        for i in tqdm(
            reversed(range(0, self.noise_scheduler.num_timesteps)),
            desc="sampling loop time step",
            total=self.noise_scheduler.num_timesteps,
        ):
            times = torch.full((batch,), i, device=device, dtype=torch.long)

            self_cond = x_start if self.net.self_cond else None
            image_embed, x_start = self.p_sample(
                image_embed,
                times,
                text_cond=text_cond,
                self_cond=self_cond,
                cond_scale=cond_scale,
            )

        if self.sampling_final_clamp_l2norm and self.predict_x_start:
            image_embed = self.l2norm_clamp_embed(image_embed)

        return image_embed

    @torch.no_grad()
    def p_sample_loop_ddim(
        self, shape, text_cond, *, timesteps, eta=1.0, cond_scale=1.0
    ):
        batch, device, alphas, total_timesteps = (
            shape[0],
            self.device,
            self.noise_scheduler.alphas_cumprod_prev,
            self.noise_scheduler.num_timesteps,
        )

        times = torch.linspace(-1.0, total_timesteps, steps=timesteps + 1)[:-1]

        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        image_embed = torch.randn(shape, device=device)

        x_start = None  # for self-conditioning

        if self.init_image_embed_l2norm:
            image_embed = l2norm(image_embed) * self.image_embed_scale

        for time, time_next in tqdm(time_pairs, desc="sampling loop time step"):
            alpha = alphas[time]
            alpha_next = alphas[time_next]

            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)

            pred = self.net.forward(
                image_embed,
                times,
                text_emb=text_cond["text_embed"],
                text_enc=text_cond["text_encodings"],
            )

            # derive x0

            if self.predict_v:
                x_start = self.noise_scheduler.predict_start_from_v(
                    image_embed, t=time_cond, v=pred
                )
            elif self.predict_x_start:
                x_start = pred
            else:
                x_start = self.noise_scheduler.predict_start_from_noise(
                    image_embed, t=time_cond, noise=pred
                )

            # clip x0 before maybe predicting noise

            if not self.predict_x_start:
                x_start.clamp_(-1.0, 1.0)

            if self.predict_x_start and self.sampling_clamp_l2norm:
                x_start = self.l2norm_clamp_embed(x_start)

            # predict noise

            pred_noise = self.noise_scheduler.predict_noise_from_start(
                image_embed, t=time_cond, x0=x_start
            )

            if time_next < 0:
                image_embed = x_start
                continue

            c1 = (
                eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            )
            c2 = ((1 - alpha_next) - torch.square(c1)).sqrt()
            noise = torch.randn_like(image_embed) if time_next > 0 else 0.0

            image_embed = x_start * alpha_next.sqrt() + c1 * noise + c2 * pred_noise

        if self.predict_x_start and self.sampling_final_clamp_l2norm:
            image_embed = self.l2norm_clamp_embed(image_embed)

        return image_embed

    @torch.no_grad()
    def p_sample_loop(self, *args, timesteps=None, **kwargs):
        timesteps = default(timesteps, self.noise_scheduler.num_timesteps)
        assert timesteps <= self.noise_scheduler.num_timesteps
        is_ddim = timesteps < self.noise_scheduler.num_timesteps

        if not is_ddim:
            normalized_image_embed = self.p_sample_loop_ddpm(*args, **kwargs)
        else:
            normalized_image_embed = self.p_sample_loop_ddim(
                *args, **kwargs, timesteps=timesteps
            )

        image_embed = self.unscale_image_embedding(normalized_image_embed)

        return image_embed

    def p_losses(self, image_embed, times, text_cond, noise=None):
        noise = default(noise, lambda: torch.randn_like(image_embed))

        image_embed_noisy = self.noise_scheduler.q_sample(
            x_start=image_embed, t=times, noise=noise
        )

        pred = self.net(
            image_embed_noisy,
            times,
            text_emb=text_cond["text_embed"],
            text_enc=text_cond["text_encodings"],
        )

        if self.predict_x_start and self.training_clamp_l2norm:
            pred = self.l2norm_clamp_embed(pred)

        if self.predict_v:
            target = self.noise_scheduler.calculate_v(image_embed, times, noise)
        elif self.predict_x_start:
            target = image_embed
        else:
            target = noise

        loss = self.noise_scheduler.loss_fn(pred, target)
        return loss

    @torch.no_grad()
    @eval_decorator
    def sample_batch_size(self, batch_size, text_cond, cond_scale=1.0):
        device = self.betas.device
        shape = (batch_size, self.image_embed_dim)

        img = torch.randn(shape, device=device)

        for i in tqdm(
            reversed(range(0, self.noise_scheduler.num_timesteps)),
            desc="sampling loop time step",
            total=self.noise_scheduler.num_timesteps,
        ):
            img = self.p_sample(
                img,
                torch.full((batch_size,), i, device=device, dtype=torch.long),
                text_cond=text_cond,
                cond_scale=cond_scale,
            )
        return img

    @torch.no_grad()
    @eval_decorator
    def sample(self, text, num_samples_per_batch=2, cond_scale=1.0, timesteps=None):
        timesteps = default(timesteps, self.sample_timesteps)

        # in the paper, what they did was
        # sample 2 image embeddings, choose the top 1 similarity, as judged by CLIP
        text = repeat(text, "b ... -> (b r) ...", r=num_samples_per_batch)

        batch_size = text.shape[0]
        image_embed_dim = self.image_embed_dim

        text_embed, text_encodings = self.language_model.embed_text(text)

        text_cond = dict(text_embed=text_embed)

        if self.condition_on_text_encodings:
            text_cond = {**text_cond, "text_encodings": text_encodings}

        image_embeds = self.p_sample_loop(
            (batch_size, image_embed_dim),
            text_cond=text_cond,
            cond_scale=cond_scale,
            timesteps=timesteps,
        )

        # retrieve original unscaled image embed

        text_embeds = text_cond["text_embed"]

        text_embeds = rearrange(
            text_embeds, "(b r) d -> b r d", r=num_samples_per_batch
        )
        image_embeds = rearrange(
            image_embeds, "(b r) d -> b r d", r=num_samples_per_batch
        )

        text_image_sims = torch.einsum(
            "b r d, b r d -> b r", l2norm(text_embeds), l2norm(image_embeds)
        )
        top_sim_indices = text_image_sims.topk(k=1).indices

        top_sim_indices = repeat(top_sim_indices, "b 1 -> b 1 d", d=image_embed_dim)

        top_image_embeds = image_embeds.gather(1, top_sim_indices)
        return rearrange(top_image_embeds, "b 1 d -> b d")

    def forward(
        self,
        text=None,
        image=None,
        text_embed=None,  # allow for training on preprocessed CLIP text and image embeddings
        image_embed=None,
        text_encodings=None,  # as well as CLIP text encodings
        *args,
        **kwargs,
    ):
        assert exists(text) ^ exists(
            text_embed
        ), "either text or text embedding must be supplied"
        assert exists(image) ^ exists(
            image_embed
        ), "either image or image embedding must be supplied"
        assert not (
            self.condition_on_text_encodings
            and (not exists(text_encodings) and not exists(text))
        ), "text encodings must be present if you specified you wish to condition on it on initialization"

        if exists(image):
            image_embed, _ = self.language_model.embed_image(image)

        # calculate text conditionings, based on what is passed in

        if exists(text):
            text_embed, text_encodings = self.language_model.embed_text(text)

        text_cond = dict(text_embed=text_embed)

        if self.condition_on_text_encodings:
            assert exists(
                text_encodings
            ), "text encodings must be present for diffusion prior if specified"
            text_cond = {**text_cond, "text_encodings": text_encodings}

        # timestep conditioning from ddpm

        batch = image_embed.shape[0]
        times = self.noise_scheduler.sample_random_times(batch)

        image_embed = self.scale_image_embedding(image_embed)

        # calculate forward loss

        return self.p_losses(image_embed, times, text_cond=text_cond, *args, **kwargs)
