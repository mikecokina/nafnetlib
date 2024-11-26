from copy import deepcopy

import numpy as np
import torch
from PIL import Image

from .utils import define_network
from ..basic_sr.base_model import BaseModel
from ..utils import img2tensor, tensor2img


class ImageRestorationModel(BaseModel):
    """Base Deblur model for single image deblur."""
    def __init__(self, opt):
        super(ImageRestorationModel, self).__init__(opt)

        # define network
        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(
                net=self.net_g,
                load_path=load_path,
                strict=True,
                param_key="params"
            )
        self.scale = int(opt['scale'])
        self.net_g.eval()

    def predict(self, image: Image.Image) -> Image.Image:
        image = image.convert('RGB')
        # noinspection PyTypeChecker
        image_arr = np.array(image)[:, :, ::-1]
        img_t = img2tensor(image_arr)
        lq = img_t.unsqueeze(dim=0).to(self.device)

        self.net_g.eval()
        with torch.no_grad():
            prediction = self.net_g(lq[0:1])
            prediction = prediction[-1].detach().cpu()

        image = tensor2img(prediction)
        image_arr = np.array(image, dtype=np.uint8)
        # noinspection PyTypeChecker
        return Image.fromarray(image_arr)
