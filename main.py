import imageio
from Generator import AttachedGSGANGenerator, CGSGANGenerator
from paths import CGSGAN_MODEL_PATH, CGSGAN_SOURCE_PATH
from Inversion import PTI

if __name__ == "__main__":

    # load images

    paths = [
        "~/projects/gan_figure_maker/id_1_out_resize/512/1.jpg",
        "~/projects/gan_figure_maker/id_1_out_resize/512/2.jpg",
        "~/projects/gan_figure_maker/id_1_out_resize/512/3.jpg",
        "~/projects/gan_figure_maker/id_1_out_resize/512/4.jpg",
        "~/projects/gan_figure_maker/id_1_out_resize/512/5.jpg",
        "~/projects/gan_figure_maker/id_1_out_resize/512/6.jpg",
        "~/projects/gan_figure_maker/id_1_out_resize/512/7.jpg",
        "~/projects/gan_figure_maker/id_1_out_resize/512/8.jpg",
        "~/projects/gan_figure_maker/id_1_out_resize/512/9.jpg",
        "~/projects/gan_figure_maker/id_1_out_resize/512/10.jpg",
        "~/projects/gan_figure_maker/id_1_out_resize/512/11.jpg",
        "~/projects/gan_figure_maker/id_1_out_resize/512/12.jpg",
        "~/projects/gan_figure_maker/id_1_out_resize/512/13.jpg",
        "~/projects/gan_figure_maker/id_1_out_resize/512/14.jpg",
    ]
    images = [imageio.imread(p) for p in paths]

    generator = CGSGANGenerator(
        model_path=CGSGAN_MODEL_PATH, source_path=CGSGAN_SOURCE_PATH, device="cuda:0"
    )
    # or use the .G attribtute
    # generator = AttachedGSGANGenerator(model=generator.G, device="cuda:0")

    pti = PTI(generator, device="cuda:0")
    images_in_bgr = pti.preprocess(images)

    pti.pre_inversion()
    weights = {"l2": 1, "lpips": 1, "id": 1}
    for _ in range(10):
        current_w, losses = pti.inversion_step(weights, lr=0.001)

    pti.pre_tuning()
    for _ in range(10):
        current_w, losses = pti.tuning_step(weights, lr=0.001)
