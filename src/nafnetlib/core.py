import abc
import os.path
from copy import deepcopy
from typing import Union, Dict

import torch

from PIL import Image

from .conf import ModelsConfiguration
from .models.restoration import ImageRestorationModel
from .utils import download_model


class AbstractNAFNetProcessor(metaclass=abc.ABCMeta):
    def __init__(self, model_id: str, model_dir: str, device: str):
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
            download_model(model_path=model_path, model_url=model_url)

    def process(self, image: Image.Image):
        return self.net.predict(image)

    @staticmethod
    def _update_opt_by_device(opt: Dict, device: Union[str, torch.device]) -> Dict:
        opt = deepcopy(opt)
        opt["num_gpu"] = 0
        if isinstance(device, torch.device):
            device = device.type
        if device == "cuda":
            opt["num_gpu"] = 1
        return opt


class DeblurProcessor(AbstractNAFNetProcessor):
    def __init__(self, model_id: str, model_dir: str, device: Union[str, torch.device]):
        super().__init__(model_id, model_dir, device)
        opt = self.model_config['gopro_width64']
        opt = self._update_opt_by_device(opt=opt, device=device)
        self.net = ImageRestorationModel(opt)


class DenoiseProcessor(AbstractNAFNetProcessor):
    def __init__(self, model_id: str, model_dir: str, device: str):
        super().__init__(model_id, model_dir, device)
