import os
import torch
import importlib


def instantiate_from_config(config, *args, **kwargs):
    return get_obj_from_str(config["target"])(
        *args, **kwargs, **config.get("params", dict())
    )


def get_obj_from_str(string, reload=False, invalidate_cache=True):
    module, cls = string.rsplit(".", 1)
    if invalidate_cache:
        importlib.invalidate_caches()
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out

    return inner


def load_stats(path: str):
    """
    Load the embedding stats from a file.
    """
    assert os.path.isfile(path), f"Could not find stats file at {path}"
    mu, std = torch.load(path, map_location="cpu")

    return mu, std
