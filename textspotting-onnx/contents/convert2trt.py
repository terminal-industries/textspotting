import onnx
from typing import Union
from DeepSolo.onnx_model import SimpleONNXReadyModel
import numpy as np
from onnxsim import simplify
import torch
import torch.onnx
import cv2
from DeepSolo.adet.config import get_cfg
from pathlib import Path
import subprocess
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Constants
OPSET_VER = 17
IMAGE_PATH = "entry_mar_14_2024_01_00_53_pm_mar_14_2024_01_01_09_pm_fps_5_frame_34.png"
INTER_ONNX = "/tmp/test_onnx_tmp.onnx"
CONFIG = "config/Base_det_export_R50.yaml"
OUTPATH = '/tmp/intermediate.onnx'
TRT_ENGINE = '/home/ubuntu/models_deliver/txtspotting_r50_v0.1.1_2K.engine'
CHECKPOINT = "/home/ubuntu/model_0319999_more_data.pth"
DIMS = (2048, 1152)
CHANNELS = 3

def setup_cfg(config_path: Union[str, Path]):
    """Set up the configuration for the model."""
    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.freeze()
    return cfg

def load_and_preprocess_image(image_path: str, dims: tuple) -> torch.Tensor:
    """Load and preprocess the image."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Failed to load img at path {image_path}")
    img = cv2.resize(img, dims)
    img_t = torch.from_numpy(img).float().permute(2, 0, 1)
    return img_t

def export_model_to_onnx(model, img_t: torch.Tensor, output_path: str, opset_version: int):
    """Export the PyTorch model to ONNX format."""
    torch.onnx.export(
        model,
        [img_t],
        output_path,
        opset_version=opset_version,
        input_names=['image'],
        output_names=['ctrl_point_cls', 'ctrl_point_coord', 'ctrl_point_text', 'bd_pointsoutput']
    )

def simplify_onnx_model(input_path: str, output_path: str):
    """Simplify the ONNX model."""
    model = onnx.load(input_path)
    model_simp, check = simplify(model)
    if not check:
        raise RuntimeError("Simplified ONNX model could not be validated")
    onnx.save(model_simp, output_path)

def convert_to_trt(onnx_path: str, trt_engine_path: str):
    """Convert the ONNX model to TensorRT engine."""
    command = [
        'polygraphy', 'convert', onnx_path,
        '-o', trt_engine_path,
        '--trt-config-script', 'trt_config.py',
        '--trt-config-func-name', 'set_trt_config'
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    logging.debug('stdout: %s', result.stdout)
    logging.debug('stderr: %s', result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to convert ONNX to TRT: {result.stderr}")

def main():
    # Set up configuration and model
    cfg = setup_cfg(CONFIG)
    model = SimpleONNXReadyModel(CONFIG, CHECKPOINT, 'cpu', width=DIMS[0], height=DIMS[1]).model
    model.eval()

    with torch.no_grad():
        # Load and preprocess image
        img_t = load_and_preprocess_image(IMAGE_PATH, DIMS)

        # Export model to ONNX
        export_model_to_onnx(model, img_t, INTER_ONNX, OPSET_VER)
        onnx.checker.check_model(INTER_ONNX)

        # Simplify ONNX model
        simplify_onnx_model(INTER_ONNX, OUTPATH)

        # Convert simplified ONNX model to TensorRT engine
        convert_to_trt(OUTPATH, TRT_ENGINE)

if __name__ == "__main__":
    main()
