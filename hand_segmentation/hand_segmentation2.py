"""Real-time hand segmentation using the Camporese PyTorch checkpoint.
https://github.com/guglielmocamporese/hands-segmentation-pytorch
https://zenodo.org/records/6964799

The supplied checkpoint is trained for *hands*, not a separate forearm/arm
class.  It returns the pixels it considers hand foreground; train or fine-tune
on masks that include arms before treating it as a hand-and-arm segmenter.

Install the model runtime once with:
    pip install torch torchvision pytorch-lightning gdown

The first run downloads the repository and its checkpoint through Torch Hub.
Press Q or Esc to close the preview.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2 as cv
import numpy as np

try:
    import torch
except ImportError:  # Let --help and importing this module work without PyTorch.
    torch = None

try:
    import gdown
except ImportError:
    gdown = None


MODEL_REPOSITORY = "guglielmocamporese/hands-segmentation-pytorch"
CHECKPOINT_ID = "1w7dztGAsPHD_fl_Kv_a8qHL4eW92rlQg"
MODEL_INPUT_SIZE = 256
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class TorchHubHandSegmenter:
    """Return a boolean foreground mask from the repository's hand model."""

    def __init__(self, device: str | None = None, threshold: float = 0.5,
                 checkpoint_path: str | Path | None = None):
        if torch is None:
            raise RuntimeError(
                "PyTorch is not installed. Run: "
                "pip install torch torchvision pytorch-lightning gdown")
        if gdown is None:
            raise RuntimeError("gdown is not installed. Run: pip install gdown")
        if not 0 < threshold < 1:
            raise ValueError("threshold must be between 0 and 1")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.threshold = threshold
        self.checkpoint_path = Path(checkpoint_path or
                                    Path(torch.hub.get_dir()) / "checkpoints" /
                                    "hands-segmentation-pytorch" / "checkpoint.ckpt")
        try:
            if not self.checkpoint_path.is_file():
                self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                gdown.download(id=CHECKPOINT_ID, output=str(self.checkpoint_path),
                               quiet=False, resume=True)
            if not self.checkpoint_path.is_file():
                raise FileNotFoundError(self.checkpoint_path)
            model = torch.hub.load(
                MODEL_REPOSITORY, "hand_segmentor", pretrained=False,
                trust_repo=True,
            )
            self.model = type(model).load_from_checkpoint(
                str(self.checkpoint_path), map_location=self.device,
                # The upstream Lightning checkpoint contains its callback state.
                # It is downloaded only from the repository's published file.
                weights_only=False,
            ).to(self.device).eval()
        except Exception as exc:
            raise RuntimeError(
                "Could not load the Torch Hub hand-segmentation model from "
                f"{self.checkpoint_path}. {type(exc).__name__}: {exc}"
            ) from exc

        self.mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)

    def segment(self, frame: np.ndarray) -> np.ndarray:
        """Return a boolean mask of the hand pixels in a BGR camera frame."""
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be a BGR image with three channels")

        height, width = frame.shape[:2]
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        rgb = cv.resize(rgb, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
                        interpolation=cv.INTER_LINEAR)
        image = torch.from_numpy(rgb).permute(2, 0, 1).contiguous().float()
        image = image.div(255).unsqueeze(0).to(self.device)
        image = (image - self.mean) / self.std
        with torch.inference_mode():
            probability = torch.softmax(self.model(image), dim=1)[0, 1]
        probability = probability.cpu().numpy()
        probability = cv.resize(probability, (width, height), interpolation=cv.INTER_LINEAR)
        return probability >= self.threshold


def draw_segmentation(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return a camera frame with predicted hand pixels overlaid in red."""
    overlay = frame.copy()
    overlay[mask] = (0, 0, 255)
    return cv.addWeighted(overlay, 0.6, frame, 0.4, 0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0, help="camera index")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--torch-device", default=None,
                        help="PyTorch device, e.g. cpu or cuda")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="hand foreground probability threshold")
    parser.add_argument("--checkpoint", type=Path,
                        help="optional local checkpoint.ckpt path")
    args = parser.parse_args(argv)

    segmenter = TorchHubHandSegmenter(args.torch_device, args.threshold,
                                      args.checkpoint)
    print("Loaded hand-only checkpoint; arms require an arm-labelled fine-tune.")
    camera = cv.VideoCapture(args.device, cv.CAP_DSHOW)
    camera.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    camera.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)
    if not camera.isOpened():
        raise RuntimeError(f"could not open camera {args.device}")
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("could not read a frame from the camera")
            cv.imshow("Torch Hub Hand Segmentation", draw_segmentation(
                frame, segmenter.segment(frame)))
            if cv.waitKey(1) & 0xFF in (27, ord("q")):
                break
    finally:
        camera.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
