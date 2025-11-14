import abc
import os.path
from copy import deepcopy
from pathlib import Path
from typing import Union, Dict

import numpy as np
import torch

from PIL import Image

from . import utils
from .conf import ModelsConfiguration
from .models.restoration import ImageRestorationModel

SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.webp', '.gif', '.tiff', '.bmp']


class AbstractNAFNetProcessor(metaclass=abc.ABCMeta):
    MODELS = []

    def __init__(self, model_id: str, model_dir: str, device: str):
        self.validate_model(model_id)

        self.device = device
        self.model_id = model_id
        self.model_dir = model_dir
        self.model_config = ModelsConfiguration(model_dir=self.model_dir)

        self.net = None
        self._download_model(self.model_id)

    def _download_model(self, model_id: str):
        config_ = self.model_config[model_id]
        model_path, model_url = config_["path"]["pretrain_network_g"], config_["model_url"]
        if not os.path.isfile(str(model_path)):
            utils.download_model(model_path=model_path, model_url=model_url)

    def process(self, image: Image.Image, tile_size: int = None, tile_overlap: int = 32) -> Image.Image:
        """
        If tile_size is None - run on the full image.
        If tile_size is set - run the model in tiles to reduce memory usage.
        """
        if tile_size is None:
            processed = self.net.predict(image)
        else:
            processed = self._process_tiled(image, tile_size=tile_size, tile_overlap=tile_overlap)

        # Garbage collection
        utils.torch_gc(self.device)

        return processed

    single = process

    def batch(self, input_dir: Union[str, Path], output_dir: Union[str, Path], progressbar=True) -> None:
        try:
            if progressbar:
                from tqdm import tqdm
            else:
                raise ImportError()
        except ImportError:
            # noinspection PyUnusedLocal
            def _tqdm(x, *args, **kwargs):
                return x

            tqdm = _tqdm

        files = utils.listdir(directory=input_dir, filter_ext=SUPPORTED_FORMATS)
        output_dir = Path(output_dir)

        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        for file_ in tqdm(files):
            file_path = Path(file_)
            output_path = output_dir / f"{file_path.stem}_processed.png"

            image = Image.open(str(file_)).convert('RGB')
            processed = self.single(image=image)
            processed.save(output_path)

    def _process_tiled(self, image: Image.Image, tile_size: int, tile_overlap: int) -> Image.Image:
        """
        Simple tiling with overlap and averaging in the overlap regions.

        tile_size - size of each tile in pixels (square tiles).
        tile_overlap - how many pixels neighboring tiles overlap.
        """
        if tile_size <= tile_overlap:
            raise ValueError("tile_size must be larger than tile_overlap")

        w, h = image.size
        stride = tile_size - tile_overlap

        # Accumulation buffers on CPU
        out = np.zeros((h, w, 3), dtype=np.float32)
        weight = np.zeros((h, w, 3), dtype=np.float32)

        # Iterate over tiles
        for top in range(0, h, stride):
            for left in range(0, w, stride):
                bottom = min(top + tile_size, h)
                right = min(left + tile_size, w)

                # Crop tile
                tile = image.crop((left, top, right, bottom))

                # Run model on tile - expects and returns PIL.Image
                pred_tile = self.net.predict(tile)
                pred_tile_np = np.asarray(pred_tile).astype(np.float32) / 255.0

                th, tw, _ = pred_tile_np.shape

                # Accumulate result and weights
                out[top:bottom, left:right, :] += pred_tile_np[: (bottom - top), : (right - left), :]
                weight[top:bottom, left:right, :] += 1.0

        # Avoid division by zero
        weight = np.maximum(weight, 1e-7)
        out /= weight

        out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(out)

    @staticmethod
    def _update_opt_by_device(opt: Dict, device: Union[str, torch.device]) -> Dict:
        opt = deepcopy(opt)
        opt["num_gpu"] = 0
        if isinstance(device, torch.device):
            device = device.type
        if device == "cuda":
            opt["num_gpu"] = 1
        return opt

    @classmethod
    def validate_model(cls, model_id) -> bool:
        if model_id not in cls.MODELS:
            raise ValueError(f"Invalid model id {model_id}, supported models are {cls.MODELS}")
        return True

    def available_models(self):
        return self.MODELS


class DeblurProcessor(AbstractNAFNetProcessor):
    MODELS = ["gopro_width64", "gopro_width32", "reds_width64"]

    def __init__(self, model_id: str, model_dir: str, device: Union[str, torch.device]):
        super().__init__(model_id, model_dir, device)
        opt = self.model_config[model_id]
        opt = self._update_opt_by_device(opt=opt, device=device)
        self.net = ImageRestorationModel(opt)


class DenoiseProcessor(AbstractNAFNetProcessor):
    MODELS = ["sidd_width64", "sidd_width32"]

    def __init__(self, model_id: str, model_dir: str, device: str):
        super().__init__(model_id, model_dir, device)
        opt = self.model_config[model_id]
        opt = self._update_opt_by_device(opt=opt, device=device)
        self.net = ImageRestorationModel(opt)


__all__ = (
    "DeblurProcessor",
    "DenoiseProcessor",
)
