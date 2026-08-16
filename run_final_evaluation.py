"""CUDA-capable end-to-end evaluation for self-contained reversible 2D-PEE."""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim

from v1_method_for_tomoka import (
    NCNNP,
    embed_two_stage_2dpee,
    extract_two_stage_2dpee,
)


CSV_FIELDS = [
    "Image",
    "EC_target",
    "Used_bits",
    "PSNR",
    "SSIM",
    "Status",
    "Payload_Match",
    "Image_Match",
    "Decoder_OK",
    "Auxiliary_bits",
    "Stage1_Stop_Rank",
    "Stage2_Stop_Rank",
    "Elapsed_seconds",
    "Error",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate reversible 2D-PEE with embedding and blind extraction."
    )
    parser.add_argument("--images-dir", default="images")
    parser.add_argument("--model-path", default="ncnnp_imagenette.pth")
    parser.add_argument("--ec", nargs="+", type=int, default=[10000, 20000])
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
    parser.add_argument("--output-root", default="results")
    return parser.parse_args()


def resolve_device(requested_device):
    if requested_device == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if requested_device == "cuda":
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return "cpu"


def make_output_dir(output_root):
    run_id = datetime.now().strftime("final_eval_%Y%m%d_%H%M%S_%f")
    output_dir = Path(output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def calculate_psnr_and_ssim(original, stego):
    mse = np.mean((original.astype(np.float32) - stego.astype(np.float32)) ** 2)
    psnr_value = 100.0 if mse == 0 else float(10 * np.log10((255 ** 2) / mse))
    ssim_value = float(ssim(original, stego, data_range=255))
    return psnr_value, ssim_value


def make_payload(seed, image_index, ec):
    payload_seed = np.random.SeedSequence([seed, image_index, ec])
    generator = np.random.default_rng(payload_seed)
    return generator.integers(0, 2, size=ec, dtype=np.uint8).tolist()


def build_model(model_path, device):
    if not model_path.is_file():
        raise FileNotFoundError(f"model checkpoint was not found: {model_path}")
    model = NCNNP().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def evaluate_case(model, image_array, image_name, image_index, ec, args, device, output_dir):
    payload = make_payload(args.seed, image_index, ec)
    row = {
        "Image": image_name,
        "EC_target": ec,
        "Used_bits": "",
        "PSNR": "",
        "SSIM": "",
        "Status": "ERROR",
        "Payload_Match": False,
        "Image_Match": False,
        "Decoder_OK": False,
        "Auxiliary_bits": "",
        "Stage1_Stop_Rank": "",
        "Stage2_Stop_Rank": "",
        "Elapsed_seconds": "",
        "Error": "",
    }

    try:
        if device == "cuda":
            torch.cuda.synchronize()
        started_at = perf_counter()
        stego, embedding_info = embed_two_stage_2dpee(
            model, image_array, payload, device, target_ec=ec
        )
        recovered, extracted_payload = extract_two_stage_2dpee(
            model, np.asarray(stego), device
        )
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed_seconds = perf_counter() - started_at

        stego_array = np.asarray(stego)
        recovered_array = np.asarray(recovered)
        payload_match = extracted_payload == payload
        image_match = np.array_equal(recovered_array, image_array)
        decoder_ok = payload_match and image_match
        psnr_value, ssim_value = calculate_psnr_and_ssim(image_array, stego_array)

        stego.save(output_dir / f"stego_2dpee_paper_EC{ec}_{image_name}")
        if not image_match:
            Image.fromarray(recovered_array).save(
                output_dir / f"recovered_EC{ec}_{image_name}"
            )

        row.update({
            "Used_bits": embedding_info["payload_length"],
            "PSNR": f"{psnr_value:.6f}",
            "SSIM": f"{ssim_value:.6f}",
            "Status": "SUCCESS" if decoder_ok else "VERIFICATION_FAILED",
            "Payload_Match": payload_match,
            "Image_Match": image_match,
            "Decoder_OK": decoder_ok,
            "Auxiliary_bits": embedding_info["auxiliary_bit_length"],
            "Stage1_Stop_Rank": embedding_info["stage1_stop_rank"],
            "Stage2_Stop_Rank": embedding_info["stage2_stop_rank"],
            "Elapsed_seconds": f"{elapsed_seconds:.6f}",
        })
        if not decoder_ok:
            row["Error"] = "payload or recovered image did not match"
    except Exception as error:
        row["Error"] = f"{type(error).__name__}: {error}"

    return row


def append_average_rows(rows, ec_values):
    for ec in ec_values:
        successes = [
            row for row in rows
            if row["EC_target"] == ec and row["Status"] == "SUCCESS"
        ]
        if not successes:
            continue
        rows.append({
            "Image": f"AVERAGE_EC{ec}",
            "EC_target": ec,
            "Used_bits": "",
            "PSNR": f"{np.mean([float(row['PSNR']) for row in successes]):.6f}",
            "SSIM": f"{np.mean([float(row['SSIM']) for row in successes]):.6f}",
            "Status": "",
            "Payload_Match": "",
            "Image_Match": "",
            "Decoder_OK": "",
            "Auxiliary_bits": "",
            "Stage1_Stop_Rank": "",
            "Stage2_Stop_Rank": "",
            "Elapsed_seconds": "",
            "Error": "",
        })


def write_run_metadata(output_dir, args, device):
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "images_dir": args.images_dir,
        "model_path": args.model_path,
        "ec": args.ec,
        "seed": args.seed,
        "device": device,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=True, indent=2)


def main():
    args = parse_args()
    if any(ec < 0 for ec in args.ec):
        raise ValueError("all EC values must be non-negative")

    device = resolve_device(args.device)
    images_dir = Path(args.images_dir)
    if not images_dir.is_dir():
        raise FileNotFoundError(f"images directory was not found: {images_dir}")

    model = build_model(Path(args.model_path), device)
    image_paths = sorted(
        path for path in images_dir.iterdir() if path.suffix.lower() == ".bmp"
    )
    if not image_paths:
        raise FileNotFoundError(f"no BMP images were found in: {images_dir}")

    output_dir = make_output_dir(args.output_root)
    write_run_metadata(output_dir, args, device)
    print(f"[Device] {device}")
    print(f"[Output] {output_dir}")
    rows = []
    for image_index, image_path in enumerate(image_paths):
        image_array = np.asarray(Image.open(image_path).convert("L"))
        for ec in args.ec:
            print(f"[Processing] {image_path.name} | EC target: {ec}")
            row = evaluate_case(
                model,
                image_array,
                image_path.name,
                image_index,
                ec,
                args,
                device,
                output_dir,
            )
            rows.append(row)
            print(
                f"  Status: {row['Status']} | Used: {row['Used_bits']}/{ec} | "
                f"PSNR: {row['PSNR']} | SSIM: {row['SSIM']}"
            )
            if row["Error"]:
                print(f"  Error: {row['Error']}")

    append_average_rows(rows, args.ec)
    csv_path = output_dir / "output_results_2dpee_paper_mapping.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    failures = [row for row in rows if row["Status"] != "SUCCESS" and not str(row["Image"]).startswith("AVERAGE_")]
    print(f"[Done] Saved to {output_dir}")
    if failures:
        print(f"[Failed] {len(failures)} case(s) did not pass verification.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
