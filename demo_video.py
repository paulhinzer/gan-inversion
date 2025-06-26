from KeyframeAnalyser import KeyFrameAnalayser
from demo_image_list import initialize_renderer
from root import get_project_path
from paths import CGSGAN_MODEL_PATH, CGSGAN_SOURCE_PATH
from Inversion import Inversion, save_image
from tqdm import tqdm
import sys

sys.path.append(CGSGAN_SOURCE_PATH)

if __name__ == "__main__":
    video_path = f"{get_project_path()}/examples/in/person_193.mp4"
    generator = initialize_renderer(CGSGAN_MODEL_PATH, "cuda:0")
    k = KeyFrameAnalayser("cuda:0")
    images = k(video_path, 5)
    i = images[0]

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
