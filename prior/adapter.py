from collections import namedtuple

import lightning.pytorch as pl
import open_clip
import torch
import torch.nn.functional as F
from resize_right import resize

EmbeddedImage = namedtuple("EmbedImageReturn", ["image_embed", "image_encodings"])
EmbeddedText = namedtuple("EmbedTextReturn", ["text_embed", "text_encodings"])


def exists(val):
    return val is not None


def l2norm(t):
    return F.normalize(t, dim=-1)


def resize_image_to(
    image, target_image_size, clamp_range=None, nearest=False, **kwargs
):
    orig_image_size = image.shape[-1]

    if orig_image_size == target_image_size:
        return image

    if not nearest:
        scale_factors = target_image_size / orig_image_size
        out = resize(image, scale_factors=scale_factors, **kwargs)
    else:
        out = F.interpolate(image, target_image_size, mode="nearest")

    if exists(clamp_range):
        out = out.clamp(*clamp_range)

    return out


class BaseClipAdapter(pl.LightningModule):
    def __init__(self, clip, **kwargs):
        super(BaseClipAdapter, self).__init__()
        self.clip = clip
        self.overrides = kwargs

    def validate_and_resize_image(self, image):
        image_size = image.shape[-1]
        assert (
            image_size >= self.image_size
        ), f"you are passing in an image of size {image_size} but CLIP requires the image size to be at least {self.image_size}"
        return resize_image_to(image, self.image_size)

    @property
    def dim_latent(self):
        raise NotImplementedError

    @property
    def image_size(self):
        raise NotImplementedError

    @property
    def image_channels(self):
        raise NotImplementedError

    @property
    def max_text_len(self):
        raise NotImplementedError

    def embed_text(self, _):
        raise NotImplementedError

    def embed_image(self, _):
        raise NotImplementedError


class OpenClipAdapter(BaseClipAdapter):
    def __init__(self, path: str):
        clip, _, preprocess = open_clip.create_model_and_transforms(model_name=path)
        self.tokenizer = open_clip.get_tokenizer(path)

        super(OpenClipAdapter, self).__init__(clip)

        self.eos_id = 49407

        text_attention_final = self.find_layer("ln_final")
        self._dim_latent = text_attention_final.weight.shape[0]

        self.handle = text_attention_final.register_forward_hook(self._hook)
        self.clip_normalize = preprocess.transforms[-1]
        self.cleared = False

    def find_layer(self, layer):
        modules = dict([*self.clip.named_modules()])
        return modules.get(layer, None)

    def clear(self):
        if self.cleared:
            return

        self.handle()

    def _hook(self, _, inputs, outputs):
        self.text_encodings = outputs

    @property
    def dim_latent(self):
        return self._dim_latent

    @property
    def image_size(self):
        image_size = self.clip.visual.image_size
        if isinstance(image_size, tuple):
            return max(image_size)
        return image_size

    @property
    def image_channels(self):
        return 3

    @property
    def max_text_len(self):
        return self.clip.context_length

    @torch.no_grad()
    def embed_text(self, text):
        text = text[..., : self.max_text_len]

        is_eos_id = text == self.eos_id
        text_mask_excluding_eos = is_eos_id.cumsum(dim=-1) == 0
        text_mask = F.pad(text_mask_excluding_eos, (1, -1), value=True)
        text_mask = text_mask & (text != 0)
        assert not self.cleared

        text_embed = self.clip.encode_text(text)
        text_encodings = self.text_encodings
        text_encodings = text_encodings.masked_fill(~text_mask[..., None], 0.0)
        del self.text_encodings
        return EmbeddedText(text_embed.float(), text_encodings.float())

    @torch.no_grad()
    def tokenize_text(self, text):
        tokens = self.tokenizer(text, context_length=self.max_text_len)
        return tokens

    @torch.no_grad()
    def embed_image(self, image):
        assert not self.cleared
        # skip the resize and normalization step
        # additionally, return the raw embedding (no normalization)
        return EmbeddedImage(self.clip.encode_image(image), None)
