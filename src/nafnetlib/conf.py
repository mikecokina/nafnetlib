from collections import OrderedDict
from pathlib import Path
from typing import Union, Dict


class _Generic(object):
    base = {
        "name": None,
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

    def _get_pretrain_network_g(self, filename: str) -> str:
        return str(Path(self.model_dir) / filename)

    def _get_model_url(self, filename: str) -> str:
        return self.model_url_prefix + filename

    def _get_base_config(self, filename: str) -> Dict:
        config_ = dict(**self.base.copy())
        config_["model_url"] = self._get_model_url(filename)
        config_["path"]["pretrain_network_g"] = self._get_pretrain_network_g(filename)
        return config_

    def _reds_width64(self):
        config_ = self._get_base_config("NAFNet-REDS-width64.pth")
        config_["name"] = "NAFNet-REDS-width64-test"
        config_["network_g"] = OrderedDict([
            ('type', 'NAFNetLocal'),
            ('width', 64),
            ('enc_blk_nums', [1, 1, 1, 28]),
            ('middle_blk_num', 1),
            ('dec_blk_nums', [1, 1, 1, 1])
        ])
        return config_

    def _gopro_width64(self):
        config_ = self._get_base_config("NAFNet-GoPro-width64.pth")
        config_["name"] = "NAFNet-GoPro-width64-test"
        config_["network_g"] = OrderedDict([
            ('type', 'NAFNetLocal'),
            ('width', 64),
            ('enc_blk_nums', [1, 1, 1, 28]),
            ('middle_blk_num', 1),
            ('dec_blk_nums', [1, 1, 1, 1])
        ])
        return config_

    def _gopro_width32(self):
        config_ = self._get_base_config("NAFNet-GoPro-width32.pth")
        config_["name"] = "NAFNet-GoPro-width32-test"
        config_["network_g"] = OrderedDict([
            ('type', 'NAFNetLocal'),
            ('width', 32),
            ('enc_blk_nums', [1, 1, 1, 28]),
            ('middle_blk_num', 1),
            ('dec_blk_nums', [1, 1, 1, 1])
        ])
        return config_

    def _sidd_width64(self):
        config_ = self._get_base_config("NAFNet-SIDD-width64.pth")
        config_["name"] = "NAFNet-SIDD-width64-test"
        config_["network_g"] = OrderedDict([
            ('type', 'NAFNet'),
            ('width', 64),
            ('enc_blk_nums', [2, 2, 4, 8]),
            ('middle_blk_num', 12),
            ('dec_blk_nums', [2, 2, 2, 2])
        ])
        return config_

    def _sidd_width32(self):
        config_ = self._get_base_config("NAFNet-SIDD-width32.pth")
        config_["name"] = "NAFNet-SIDD-width32-test"
        config_["network_g"] = OrderedDict([
            ('type', 'NAFNet'),
            ('width', 32),
            ('enc_blk_nums', [2, 2, 4, 8]),
            ('middle_blk_num', 12),
            ('dec_blk_nums', [2, 2, 2, 2])
        ])
        return config_

    def _get_model_config(self, model_id: str):
        model_config_fn = getattr(self, f'_{model_id}')
        if model_config_fn is None:
            raise ValueError(f"Invalid model id {model_id}")
        return model_config_fn()

    def __getitem__(self, model_id: str):
        return self._get_model_config(model_id)
