import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from typing import List, Tuple

import torch
from PIL import Image
import hydra
import gradio as gr
from omegaconf import DictConfig
from torchvision import transforms as T
from torchvision import transforms
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import requests
import utils
import urllib
log = utils.get_pylogger(__name__)

def demo(cfg: DictConfig) -> Tuple[dict, dict]:
    """Demo function.
    Args:
        cfg (DictConfig): Configuration composed by Hydra.
    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    assert cfg.ckpt_path
    log.info("Running Demo")
    log.info(f"Instantiating scripted model <{cfg.ckpt_path}>")
    model = torch.jit.load(cfg.ckpt_path)
    log.info(f"Loaded Model: {model}")
    ##response = requests.get("https://git.io/JJkYN")
    labels_dict =  { 0:"airplane",
                    1:"automobile",
                    2:"bird",
                    3:"cat",
                    4: "deer",
                    5: "dog",
                    6: "frog",
                    7:"horse",
                    8:"ship",
                    9:"truck"}
 
    def recognize_image(image):
        if image is None:
            return None
        ##inp = transforms.ToTensor()(image).unsqueeze(0)
        config = resolve_data_config({}, model=model)
        transform = create_transform(**config)
        tensor = transform(image).unsqueeze(0)
        preds = model.forward_jit(tensor)
        preds = preds[0].tolist()
        return {str(labels_dict[i]): preds[i] for i in range(10)}

    demo = gr.Interface(
        fn=recognize_image,
        inputs=gr.Image(type="pil"),
        outputs=[gr.Label(num_top_classes=3)]
    )
    demo.launch(server_name="0.0.0.0", share= True)

@hydra.main(
    version_base="1.2", config_path=root / "configs", config_name="demo_scripted.yaml"
)
def main(cfg: DictConfig) -> None:
    demo(cfg)

if __name__ == "__main__":
    main()