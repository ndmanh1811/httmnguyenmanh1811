"""
train_ppe.py
------------
Quy trình Huấn luyện & Tinh chỉnh (Fine-tuning) Model PPE YOLOv8 Chuẩn Công Nghiệp.
Giải quyết triệt để vấn đề nhận diện nhầm khói, hơi nước, vật thể thành Người / Vi phạm PPE.

Nguyên nhân cốt lõi gây nhận nhầm ở model cũ (30 epochs):
  1. Thiếu Negative Background Samples (Ảnh nền âm):
     Bộ dữ liệu chỉ toàn ảnh có người/bảo hộ, không có ảnh khói/hơi nước/xưởng trống.
     -> Model bị ép phải tìm ra vật thể, nên khói cuộn biến thành 'Person' hoặc 'NO-Hardhat'.
  2. Số epoch quá ít (30 epochs):
     Mạng YOLOv8 chưa kịp hội tụ để phân biệt ranh giới texture phức tạp giữa khói và người.

Giải pháp đạt độ chính xác cao (Near-Zero False Positive):
  - Thêm 10-15% ảnh nền âm tính (ảnh khói, lửa, xưởng trống kèm file .txt trống rỗng).
  - Train 80-120 epochs với early stopping (patience=20).
  - Khởi tạo từ checkpoint 'yolov8s.pt' (cân bằng hoàn hảo tốc độ FPS và độ chính xác).

Cách sử dụng:
  py train_ppe.py --data path/to/data.yaml --epochs 100 --bg-dir path/to/smoke_or_empty_images
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models_dir"
TARGET_MODEL_PATH = MODELS_DIR / "best_hardhat.pt"


def add_background_negative_samples(
    dataset_images_dir: Path,
    dataset_labels_dir: Path,
    bg_sources_dir: Path,
    max_samples: int = 150,
) -> int:
    """
    Tự động thêm ảnh nền âm tính (Negative Background Images) vào tập dữ liệu.
    Mỗi ảnh âm tính sẽ được tạo một file .txt rỗng (0 bytes).
    Theo chuẩn Ultralytics YOLO, file nhãn rỗng dạy mạng nơ-ron nhận diện 'Đây là nền/khói/không có người'.
    """
    if not bg_sources_dir.is_dir():
        print(f"[!] Thư mục ảnh nền âm tính không tồn tại: {bg_sources_dir}")
        return 0

    dataset_images_dir.mkdir(parents=True, exist_ok=True)
    dataset_labels_dir.mkdir(parents=True, exist_ok=True)

    supported_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    bg_files: list[Path] = []
    for ext in supported_exts:
        bg_files.extend(bg_sources_dir.glob(ext))

    if not bg_files:
        print(f"[!] Không tìm thấy file ảnh nào trong {bg_sources_dir}")
        return 0

    added_count = 0
    for idx, img_path in enumerate(bg_files[:max_samples]):
        target_img_name = f"negative_bg_{idx:04d}_{img_path.name}"
        target_img = dataset_images_dir / target_img_name
        target_lbl = dataset_labels_dir / f"{target_img.stem}.txt"

        # Copy ảnh vào thư mục images của dataset
        shutil.copy2(img_path, target_img)
        # Tạo file nhãn 0-byte (empty label)
        with open(target_lbl, "w", encoding="utf-8") as f:
            pass

        added_count += 1

    print(f"[+] Đã thêm {added_count} ảnh nền âm tính (kèm file .txt rỗng) vào {dataset_images_dir.name}")
    return added_count


def extract_frames_from_video(
    video_path: Path,
    output_dir: Path,
    frame_interval: int = 25,
    max_frames: int = 80,
) -> int:
    """Trích xuất frame từ video khói/lửa/xưởng trống để làm negative samples."""
    import cv2

    if not video_path.is_file():
        print(f"[!] Không tìm thấy video: {video_path}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[!] Không thể mở video: {video_path}")
        return 0

    frame_idx = 0
    saved_count = 0
    stem = video_path.stem

    while True:
        ret, frame = cap.read()
        if not ret or saved_count >= max_frames:
            break
        if frame_idx % frame_interval == 0:
            out_file = output_dir / f"bg_{stem}_f{frame_idx:05d}.jpg"
            cv2.imwrite(str(out_file), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            saved_count += 1
        frame_idx += 1

    cap.release()
    print(f"[+] Đã trích xuất {saved_count} frames nền từ {video_path.name} -> {output_dir}")
    return saved_count


def validate_dataset(data_yaml_path: Path) -> dict:
    """Kiểm tra tính hợp lệ của dataset và đếm tỷ lệ ảnh âm tính."""
    import yaml

    if not data_yaml_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình dataset: {data_yaml_path}")

    with open(data_yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print("\n--- BÁO CÁO KIỂM TRA DATASET ---")
    print(f"File data.yaml: {data_yaml_path}")
    print(f"Số lượng lớp (nc): {config.get('nc')}")
    print(f"Danh sách lớp (names): {config.get('names')}")

    # Đếm số lượng ảnh train
    train_path_raw = config.get("train")
    if train_path_raw:
        train_dir = Path(train_path_raw)
        if not train_dir.is_absolute():
            train_dir = data_yaml_path.parent / train_dir

        if train_dir.exists():
            img_files = list(train_dir.glob("*.[jJ][pP][gG]")) + list(train_dir.glob("*.[pP][nN][gG]"))
            total_imgs = len(img_files)

            # Đếm số file nhãn rỗng (negative samples)
            lbl_dir = train_dir.parent / "labels"
            if not lbl_dir.is_dir():
                lbl_dir = Path(str(train_dir).replace("images", "labels"))

            neg_count = 0
            if lbl_dir.is_dir():
                for img in img_files:
                    lbl = lbl_dir / f"{img.stem}.txt"
                    if lbl.is_file() and lbl.stat().st_size == 0:
                        neg_count += 1

            pct = (neg_count / total_imgs * 100) if total_imgs > 0 else 0
            print(f"Tổng số ảnh train: {total_imgs}")
            print(f"Số ảnh nền âm tính (negative background): {neg_count} ({pct:.1f}%)")
            if pct < 8.0:
                print("⚠️ [CẢNH BÁO] Tỷ lệ ảnh âm tính < 8%. Nên thêm ảnh khói/xưởng trống để triệt tiêu báo động giả!")
            else:
                print("✅ [TỐT] Tỷ lệ ảnh âm tính đạt chuẩn (8 - 15%), giúp triệt tiêu ảo giác khói/vật thể.")

    print("--------------------------------\n")
    return config


def train(
    data_yaml: str,
    base_weights: str = "yolov8s.pt",
    epochs: int = 100,
    batch_size: int = 16,
    imgsz: int = 640,
    patience: int = 20,
    device: str = "",
    project: str = "ppe_training",
    name: str = "yolov8s_ppe_v2",
) -> Path:
    """Chạy quy trình huấn luyện YOLOv8 chuẩn công nghiệp."""
    data_path = Path(data_yaml).resolve()
    validate_dataset(data_path)

    print(f"[*] Bắt đầu huấn luyện mô hình YOLOv8:")
    print(f"    - Pretrained weights: {base_weights}")
    print(f"    - Epochs: {epochs}")
    print(f"    - Batch size: {batch_size}")
    print(f"    - Image size: {imgsz}")
    print(f"    - Early stopping patience: {patience}")
    print(f"    - Device: {device or 'Auto detect (GPU CUDA ưu tiên)'}")

    model = YOLO(base_weights)

    import torch

    selected_device = device if device else (0 if torch.cuda.is_available() else "cpu")
    train_args = {
        "data": str(data_path),
        "epochs": epochs,
        "batch": batch_size,
        "imgsz": imgsz,
        "patience": patience,
        "save": True,
        "save_period": 10,
        "device": selected_device,
        "project": project,
        "name": name,
        "exist_ok": True,
        "mosaic": 1.0,
        "mixup": 0.1,
        "fliplr": 0.5,
        "degrees": 5.0,
        "verbose": True,
    }

    results = model.train(**train_args)

    best_ckpt = Path(project) / name / "weights" / "best.pt"
    if best_ckpt.is_file():
        print(f"\n[+] Huấn luyện thành công! File weights tốt nhất: {best_ckpt}")

        # Tự động sao lưu và cập nhật model nếu chạy cục bộ
        try:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            if TARGET_MODEL_PATH.is_file():
                backup_path = MODELS_DIR / "best_hardhat_backup.pt"
                shutil.copy2(TARGET_MODEL_PATH, backup_path)
                print(f"[*] Đã sao lưu model cũ -> {backup_path}")

            shutil.copy2(best_ckpt, TARGET_MODEL_PATH)
            print(f"✅ Đã triển khai weights mới vào hệ thống: {TARGET_MODEL_PATH}")
        except Exception as e:
            print(f"[*] [Kaggle/Colab] Bạn có thể tải file weights tại: {best_ckpt.resolve()} ({e})")
    else:
        print("[!] Không tìm thấy file best.pt sau khi train.")

    return best_ckpt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Huấn luyện mô hình PPE YOLOv8 chống báo động giả (Near-Zero False Alarm)"
    )
    parser.add_argument("--data", type=str, default="", help="Đường dẫn tới file data.yaml của dataset")
    parser.add_argument("--base", type=str, default="yolov8s.pt", help="Checkpoint gốc (mặc định yolov8s.pt)")
    parser.add_argument("--epochs", type=int, default=100, help="Số epochs huấn luyện (khuyến nghị 80 - 120)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (8, 16 hoặc 32 tùy GPU VRAM)")
    parser.add_argument("--imgsz", type=int, default=640, help="Kích thước ảnh đầu vào (mặc định 640)")
    parser.add_argument("--patience", type=int, default=20, help="Patience cho Early Stopping (mặc định 20)")
    parser.add_argument("--device", type=str, default="", help="Device GPU (vd: '0' hoặc 'cpu')")
    parser.add_argument("--bg-dir", type=str, default="", help="Thư mục chứa ảnh nền âm tính (khói, lửa, xưởng trống)")
    parser.add_argument("--extract-video-bg", type=str, default="", help="Đường dẫn video để trích xuất ảnh nền âm tính")

    args = parser.parse_args()

    # Nếu truyền video để trích xuất background
    if args.extract_video_bg:
        v_path = Path(args.extract_video_bg)
        out_bg = BASE_DIR / "negative_bg_samples"
        extract_frames_from_video(v_path, out_bg)
        if not args.bg_dir:
            args.bg_dir = str(out_bg)

    if not args.data:
        print("=" * 70)
        print("HƯỚNG DẪN HUẤN LUYỆN MODEL PPE ĐẠT CHUẨN CÔNG NGHIỆP (NEAR-ZERO FALSE ALARM)")
        print("=" * 70)
        print("1. Chuẩn bị dataset PPE (format YOLOv8 data.yaml)")
        print("2. Thu thập ảnh khói, lửa, hơi nước, xưởng vắng người (để làm Negative Background)")
        print("3. Chạy lệnh tự động train:")
        print("     py train_ppe.py --data path/to/data.yaml --epochs 100 --bg-dir path/to/smoke_images")
        print("   Hoặc trích xuất frame trực tiếp từ video khói chay2.mp4:")
        print("     py train_ppe.py --extract-video-bg static/uploads/1789913139_chay2.mp4 --data path/to/data.yaml")
        print("=" * 70)
        return

    # Nếu có chỉ định thư mục ảnh nền âm tính và data.yaml
    if args.bg_dir and args.data:
        data_yaml = Path(args.data).resolve()
        if data_yaml.is_file():
            import yaml
            with open(data_yaml, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            train_dir_str = cfg.get("train", "")
            if train_dir_str:
                train_img_dir = Path(train_dir_str)
                if not train_img_dir.is_absolute():
                    train_img_dir = data_yaml.parent / train_img_dir
                train_lbl_dir = Path(str(train_img_dir).replace("images", "labels"))
                add_background_negative_samples(train_img_dir, train_lbl_dir, Path(args.bg_dir))

    train(
        data_yaml=args.data,
        base_weights=args.base,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        patience=args.patience,
        device=args.device,
    )


if __name__ == "__main__":
    main()
