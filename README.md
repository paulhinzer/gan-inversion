# Gan Inversion

Inversion Utilities and Interfaces for Generative Adversarial Networks.

## Getting Started

To obtain the models and build the neccecary preprocessing utilities, use `./install.sh`. It creates a new conda environment named `ganinv`. If your cuda version is not compliant with the version in the `install.sh`, follow the steps manually and ensure all the packages from `requirements.txt` are installed in your environment.

```sh
./install.sh
conda activate ganinv
```

Then, optionally, if the Generator is not attached to a running CGS-GAN, write a new file `paths.py` with the following contents:

```py
CGSGAN_SOURCE_PATH = "<PATH_TO_CGSGAN_SOURCE>"
CGSGAN_MODEL_PATH = "<PATH_TO_CGSGAN_MODEL>"
```

See `./demo_image_list.py` and `./demo_video.py` for examples. The paths might have to be changed to existing paths on your machine.

