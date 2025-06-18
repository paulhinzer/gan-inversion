# Gan Inversion

Inversion Utilities and Interfaces for Generative Adversarial Networks.

## Getting Started

To obtain the models and build the neccecary preprocessing utilities, run `./install.sh`. Then, optionally, if the Generator is not attached to a running CGS-GAN, write a new file `paths.py` with the following contents:

```py
CGSGAN_SOURCE_PATH = "<PATH_TO_CGSGAN_SOURCE>"
CGSGAN_MODEL_PATH = "<PATH_TO_CGSGAN_MODEL>"
```

See `main.py` for an example. The paths might have to be changed to existing paths on your machine.

