"""
export_tensorrt.py - Utility to export YOLO detection and pose models to TensorRT FP16 engine.

Usage:
    py -3.13 export_tensorrt.py --model ppe --half
    py -3.13 export_tensorrt.py --model fall --half
    py -3.13 export_tensorrt.py --model fire --half
    py -3.13 export_tensorrt.py --model path/to/model.pt --imgsz 640 --half

Phase 2 Architecture Target:
    - Target: High-throughput NVIDIA GPU inference (e.g. RTX 40-series, Jetson Orin)
    - Export format: TensorRT Engine (.engine) with FP16 precision
    - Fallback: PyTorch CUDA / CPU float32 if TensorRT runtime not present.
"""

import argparse
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("export_tensorrt")

MODEL_ALIASES = {
    "ppe": os.path.join(os.path.dirname(__file__), "models_dir", "best_hardhat.pt"),
    "fall": os.path.join(os.path.dirname(__file__), "models_dir", "yolo26s-pose.pt"),
    "fire": os.path.join(os.path.dirname(__file__), "models_dir", "fire_smoke_yolo26s.pt"),
    "fire_v8": os.path.join(os.path.dirname(__file__), "models_dir", "fire_smoke_yolov8n.pt"),
}


def check_environment() -> dict:
    """Check hardware and library prerequisites for TensorRT export."""
    status = {
        "cuda_available": False,
        "device_name": "None",
        "tensorrt_installed": False,
        "torch_version": "",
    }
    try:
        import torch
        status["torch_version"] = torch.__version__
        status["cuda_available"] = torch.cuda.is_available()
        if status["cuda_available"]:
            status["device_name"] = torch.cuda.get_device_name(0)
    except Exception as e:
        logger.warning("Torch check failed: %s", e)

    try:
        import tensorrt
        status["tensorrt_installed"] = True
        status["tensorrt_version"] = tensorrt.__version__
    except ImportError:
        status["tensorrt_installed"] = False

    return status


def export_model(
    model_path: str,
    imgsz: int = 640,
    half: bool = True,
    batch: int = 1,
    device: int = 0,
    workspace: int = 4,
) -> str:
    """
    Export a YOLO .pt checkpoint to TensorRT .engine format.

    Args:
        model_path: Path to .pt file or model alias ('ppe', 'fall', 'fire')
        imgsz: Inference input resolution (default 640)
        half: Enable FP16 half precision (recommended for RTX GPUs)
        batch: Max batch size (default 1 for stream inference)
        device: CUDA device index (default 0)
        workspace: TensorRT workspace memory limit in GB (default 4)

    Returns:
        Path to exported .engine file.
    """
    resolved_path = MODEL_ALIASES.get(model_path.lower(), model_path)
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"Model file not found: {resolved_path}")

    env = check_environment()
    logger.info("Pre-flight Environment Check:")
    logger.info("  - PyTorch: %s", env["torch_version"])
    logger.info("  - CUDA Available: %s (Device: %s)", env["cuda_available"], env["device_name"])
    logger.info("  - TensorRT Installed: %s", env["tensorrt_installed"])

    if not env["cuda_available"]:
        msg = (
            "TensorRT export requires an active NVIDIA GPU with CUDA enabled in PyTorch.\n"
            "Current environment is running on CPU. Please run on a CUDA-enabled machine:\n"
            "  1. Ensure NVIDIA GPU driver & CUDA toolkit are installed.\n"
            "  2. Install CUDA PyTorch: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121\n"
            "  3. Install TensorRT: pip install tensorrt\n"
        )
        logger.error(msg)
        raise RuntimeError(msg)

    from ultralytics import YOLO

    logger.info("Loading model weights from %s ...", resolved_path)
    model = YOLO(resolved_path)

    logger.info(
        "Starting TensorRT export (imgsz=%d, half=%s, batch=%d, workspace=%dGB, device=%d) ...",
        imgsz, half, batch, workspace, device
    )

    # Ultralytics export to engine
    engine_path = model.export(
        format="engine",
        imgsz=imgsz,
        half=half,
        batch=batch,
        device=device,
        workspace=workspace,
        dynamic=False,
        simplify=True,
    )

    logger.info("Export successful! TensorRT engine generated at: %s", engine_path)
    return engine_path


def main():
    parser = argparse.ArgumentParser(description="Export YOLO models to TensorRT FP16 Engine")
    parser.add_argument(
        "--model",
        type=str,
        default="ppe",
        help="Model path (.pt) or alias ('ppe', 'fall', 'fire', 'fire_v8')",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Input image dimension (default: 640)")
    parser.add_argument("--half", action="store_true", default=True, help="Use FP16 precision (default: True)")
    parser.add_argument("--batch", type=int, default=1, help="Max batch size (default: 1)")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index (default: 0)")
    parser.add_argument("--workspace", type=int, default=4, help="TensorRT workspace memory in GB (default: 4)")
    parser.add_argument("--check-only", action="store_true", help="Only check hardware environment and exit")

    args = parser.parse_args()

    if args.check_only:
        env = check_environment()
        print("\n=== Environment Status ===")
        for k, v in env.items():
            print(f"  {k}: {v}")
        return

    try:
        export_model(
            model_path=args.model,
            imgsz=args.imgsz,
            half=args.half,
            batch=args.batch,
            device=args.device,
            workspace=args.workspace,
        )
    except Exception as e:
        logger.error("TensorRT export failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
