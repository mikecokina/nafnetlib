from collections import OrderedDict
from pathlib import Path
from typing import Union


class _Generic(object):
    base = {
        "model_url": None,
        "model_file": None,
        "model_type": None,
        "scale": 1,
        "num_gpu": 0,
        "manual_seed": 10,
        "network_g": None,
        "path": {
            "pretrain_network_g": None,
            "strict_load_g": True,
            "resume_state": None,
        },
        "is_train": False,
        "dist": False,
    }


class ModelsConfiguration(_Generic):
    def __init__(self, model_dir: Union[str, Path]):
        if isinstance(model_dir, str):
            model_dir = Path(model_dir)
        self.model_dir = model_dir
        self.model_url_prefix = "https://huggingface.co/mikestealth/nafnet-models/resolve/main/"

    def _gopro_width64(self):
        config_ = dict(**self.base.copy())
        config_["model_url"] = self.model_url_prefix + "NAFNet-REDS-width64.pth"
        config_["path"]["pretrain_network_g"] = str(Path(self.model_dir) / "NAFNet-REDS-width64.pth")
        config_['network_g'] = OrderedDict([
            ('type', 'NAFNetLocal'),
            ('width', 64),
            ('enc_blk_nums', [1, 1, 1, 28]),
            ('middle_blk_num', 1),
            ('dec_blk_nums', [1, 1, 1, 1])]
        )
        return config_

    def _reds_width64(self):
        config_ = self.base.copy()
        return config_

    def _get_model_config(self, model_id: str):
        model_config_fn = getattr(self, f'_{model_id}')
        if model_config_fn is None:
            # TODO: update existing and add more comprehnsive message
            raise ValueError("Invalid model id")
        return model_config_fn()

    def __getitem__(self, model_id: str):
        return self._get_model_config(model_id)
