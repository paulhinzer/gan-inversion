import imageio
from Generator import AttachedGSGANGenerator, CGSGANGenerator
from paths import CGSGAN_MODEL_PATH, CGSGAN_SOURCE_PATH
from Inversion import Inversion

if __name__ == "__main__":

    # load images

    paths = [
        "~/projects/gan_figure_maker/id_1_out_resize/512/1.jpg",
        "~/projects/gan_figure_maker/id_1_out_resize/512/2.jpg",
        "~/projects/gan_figure_maker/id_1_out_resize/512/3.jpg",
        "~/projects/gan_figure_maker/id_1_out_resize/512/4.jpg",
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
    weights = {"l2": 1, "lpips": 1, "id": 1}
    for _ in range(10):
        current_w, losses = inverter.inversion_step(weights, lr=0.001)

    inverter.pre_tuning()
    for _ in range(10):
        current_w, losses = inverter.tuning_step(weights, lr=0.001)
