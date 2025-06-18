import imageio
from Generator import AttachedGSGANGenerator, CGSGANGenerator
from paths import CGSGAN_MODEL_PATH, CGSGAN_SOURCE_PATH
from Inversion import Inversion
from tqdm import tqdm

if __name__ == "__main__":

    # load images

    paths = [
        "~/projects/gan_figure_maker/id_1_out_resize/512/1.jpg",
        # "~/projects/gan_figure_maker/id_1_out_resize/512/2.jpg",
        # "~/projects/gan_figure_maker/id_1_out_resize/512/3.jpg",
        # "~/projects/gan_figure_maker/id_1_out_resize/512/4.jpg",
    ]
    images = [imageio.imread(p) for p in paths]

    generator = CGSGANGenerator(
        model_path=CGSGAN_MODEL_PATH, source_path=CGSGAN_SOURCE_PATH, device="cuda:0"
    )
    # or use the .G attribtute
    generator = AttachedGSGANGenerator(model=generator.G, device="cuda:0")

    inverter = Inversion(generator, device="cuda:0")
    images_in_bgr = inverter.preprocess(images)

    inverter.pre_inversion()
    weights = {"mse_loss": 1, "lpips_loss": 1, "id_loss": 1, "lr": 0.001}
    for _ in tqdm(range(10)):
        current_w, losses = inverter.inversion_step(weights)

    inverter.pre_tuning()
    for _ in range(10):
        current_w, losses = inverter.tuning_step(weights)
