import imageio
from paths import CGSGAN_MODEL_PATH, CGSGAN_SOURCE_PATH
from Inversion import Inversion, save_image
from tqdm import tqdm
import sys
import pickle
import copy


sys.path.append(CGSGAN_SOURCE_PATH)


def initialize_renderer(model_path, device):
    with open(model_path, "rb") as file:
        save_file = pickle.load(file)
    G = save_file["G_ema"].to(device)
    G = copy.deepcopy(G).eval().requires_grad_(True).to(device)
    return G


if __name__ == "__main__":
    paths = [
        "~/projects/gan_figure_maker/id_1_out_resize/512/1.jpg",
        "~/projects/gan_figure_maker/id_1_out_resize/512/2.jpg",
        "~/projects/gan_figure_maker/id_1_out_resize/512/3.jpg",
        "~/projects/gan_figure_maker/id_1_out_resize/512/4.jpg",
    ]
    images = [imageio.imread(p) for p in paths]
    i = images[0]
    generator = initialize_renderer(CGSGAN_MODEL_PATH, "cuda:0")

    inverter = Inversion(generator, device="cuda:0")
    images_in_bgr = inverter.preprocess(images, target_size=512)

    weights = {
        "mse_loss": 1,
        "lpips_loss": 1,
        "id_loss": 1,
        "lr": 0.001,
        "batch_size": 3,
    }
    for _ in tqdm(range(400)):
        current_w, losses = inverter.inversion_step(weights)
    image = inverter.render_w()
    save_image(image, name="image_after_inversion")
    for _ in tqdm(range(100)):
        current_w, losses = inverter.tuning_step(weights)
    image = inverter.render_w()
    save_image(image, name="image_after_tuning")
