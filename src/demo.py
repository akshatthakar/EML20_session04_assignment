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
from src import utils
from torchvision import transforms

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

 
    def recognize_image(image):
        if image is None:
            return None
        inp = transforms.ToTensor()(image).unsqueeze(0)
        preds = model.forward_jit(inp)
        preds = preds[0].tolist()
        return {str(i): preds[i] for i in range(10)}

    demo = gr.Interface(
        fn=recognize_image,
        inputs=gr.Image(type="pil"),
        outputs=[gr.Label(num_top_classes=10)],
        live=True,
    )
    demo.launch(server_name="0.0.0.0")

@hydra.main(
    version_base="1.2", config_path=root / "configs", config_name="demo_scripted.yaml"
)
def main(cfg: DictConfig) -> None:
    demo(cfg)

if __name__ == "__main__":
    main()