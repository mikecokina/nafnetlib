import abc
import os.path
from copy import deepcopy
from pathlib import Path
from typing import Union, Dict

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

    def process(self, image: Image.Image) -> Image.Image:
        return self.net.predict(image)

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
            deblured = self.single(image=image)
            deblured.save(output_path)

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
