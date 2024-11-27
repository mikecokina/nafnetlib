[![GitHub version](https://img.shields.io/badge/version-0.1.0-yellow.svg)](https://github.com/mikecokina/nafnetlib)
[![Licence GPLv2](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/license/mit)
[![Python version](https://img.shields.io/badge/python-3.10|3.11-orange.svg)](https://www.python.org/doc/versions/)
[![OS](https://img.shields.io/badge/os-Linux|Windows|macOS-magenta.svg)](https://www.gnu.org/gnu/linux-and-gnu.html)



# NAFNet Image Restoration
This repository contains implementations of the NAFNet (Nonlinear Activation Free Network) for image restoration tasks such as deblurring and denoising. The core of the project includes a modular framework for easy processing of images using pre-trained models.


## Features
- **Image Restoration:** The core functionality includes deblurring and denoising using state-of-the-art models.
- **Modular Design:** Separate processors for different tasks, with a base class for common functionality.
- **Model Management:** Automatically downloads and manages model files for easy usage.

## Installation
Via PyPI
You can easily install the nafnetlib package from PyPI:
`pip install nafnetlib`

### Requirements
- Python 3.x
- PyTorch
- PIL (Pillow)
- Other dependencies listed in requirements.txt

You can install the dependencies using pip:

`pip install -r requirements.txt`

These dependencies will be automatically installed when you install `nafnetlib` via pip.

## Usage
Initialization and Image Processing
To use the deblurring or denoising functionality, you can initialize the corresponding processor class (DeblurProcessor or DenoiseProcessor), and then use the process() method to process images.

Here’s an example of how to use the `DeblurProcessor` and `DenoiseProcessor`:

```python
from PIL import Image
from nafnetlib import DeblurProcessor, DenoiseProcessor

# Initialize the deblurring processor
db_processor = DeblurProcessor(
    model_id="reds_width64",
    model_dir="/absolute/path/to/model/directory",
    device="cuda"
)

# Print available models
print(db_processor.available_models())

# Initialize the denoising processor
db_processor = DenoiseProcessor(
    model_id="sidd_width64",
    model_dir="/absolute/path/to/model/directory",
    device="cuda"
)

# Load image for processing
img_path = "path/to/image/file"
image = Image.open(img_path)

# Process the image (e.g., deblur or denoise)
db_processor.process(image).show()
```

## Supported Models
The following models are available for different image restoration tasks:

#### Image Deblurring (GoPro):
Models: `gopro_width64`, `gopro_width32`

#### Image Denoising (SIDD):
Models: `sidd_width64`, `sidd_width32`

#### Image Deblurring with JPEG Artifacts (REDS):
Models: `reds_width64`

These models are derived from the `megvii-research/nafnet` GitHub repository.

## Not-supported Models
`Baseline-GoPro-*`, `Baseline-SIDD-*` and `NAFSSR-L_*` 

## Model Download
When initializing the processors, models are automatically downloaded if they are not found in the specified directory. 
Supported models come from `megvii-research/nafnet`.

## Contributing
Feel free to fork the repository and submit issues or pull requests. Contributions are welcome!

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Citations
If NAFNet helps your research or work, please consider citing NAFNet.

```
@article{chen2022simple,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2204.04676},
  year={2022}
}
```

If NAFSSR helps your research or work, please consider citing NAFSSR.

```
@InProceedings{chu2022nafssr,
    author    = {Chu, Xiaojie and Chen, Liangyu and Yu, Wenqing},
    title     = {NAFSSR: Stereo Image Super-Resolution Using NAFNet},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2022},
    pages     = {1239-1248}
}
```