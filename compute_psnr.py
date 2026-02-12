import math
import os
import sys
from typing import List

import torch
from torch.autograd import Variable

from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model


def _tensor_to_01(t: torch.Tensor) -> torch.Tensor:
    """Convert normalized [-1, 1] tensor to [0, 1]."""
    return (t + 1.0) / 2.0


def _psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute PSNR for a single image tensor in [0, 1]."""
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def _shape_hw_chw(t: torch.Tensor) -> str:
    """Pretty-print tensor shape as HxW for CHW tensors."""
    if t.dim() == 3:
        return f"{t.shape[1]}x{t.shape[2]}"
    if t.dim() >= 2:
        return f"{t.shape[-2]}x{t.shape[-1]}"
    return str(tuple(t.shape))


def _compute_psnr_for_phase(opt, phase: str) -> None:
    opt.phase = phase
    opt.nThreads = max(0, int(opt.nThreads))
    opt.batchSize = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.use_encoded_image = True  # ensures test_B is loaded for PSNR

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()

    model = create_model(opt)
    if opt.data_type == 16:
        model.half()
    elif opt.data_type == 8:
        model.type(torch.uint8)

    psnrs: List[float] = []
    compared_count = 0
    mismatched_count = 0

    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break

        if opt.data_type == 16:
            data['label'] = data['label'].half()
            data['inst'] = data['inst'].half()
            data['image'] = data['image'].half()
        elif opt.data_type == 8:
            data['label'] = data['label'].uint8()
            data['inst'] = data['inst'].uint8()
            data['image'] = data['image'].uint8()

        # Standard pix2pixHD mapping: A -> B
        # For label_nc == 0, A = train/test_A (data['label']), B = train/test_B (data['image'])
        input_tensor = data['label']
        target_tensor = data['image']

        with torch.no_grad():
            generated = model.inference(input_tensor, data['inst'], data['image'])

        pred = _tensor_to_01(generated.data[0].float().cpu())
        target = _tensor_to_01(target_tensor[0].float().cpu())

        same_shape = pred.shape == target.shape

        img_path = data['path'][0] if isinstance(data['path'], list) else data['path']
        pred_hw = _shape_hw_chw(pred)
        target_hw = _shape_hw_chw(target)

        if not same_shape:
            mismatched_count += 1
            print(
                f"[{phase}] [{i+1}] SIZE_MISMATCH | gen={pred_hw} target={target_hw} | {img_path}"
            )
            continue

        psnr_val = _psnr(pred, target)
        psnrs.append(psnr_val)
        compared_count += 1
        print(
            f"[{phase}] [{i+1}] SIZE_MATCH | gen={pred_hw} target={target_hw} | "
            f"PSNR: {psnr_val:.4f} | {img_path}"
        )

    print(
        f"[{phase}] Compared: {compared_count}, Skipped(size mismatch): {mismatched_count}, "
        f"Total seen: {compared_count + mismatched_count}"
    )

    if psnrs:
        avg = sum(psnrs) / len(psnrs)
        print(f"[{phase}] Average PSNR over {len(psnrs)} images: {avg:.4f}")
    else:
        print(
            f"[{phase}] No PSNR computed. Check your {phase}_* folders/options, "
            "or ensure generated and target image sizes match."
        )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--phases", type=str, default="test")
    phase_args, remaining = parser.parse_known_args()

    # Remove custom args before TestOptions parses argv.
    original_argv = sys.argv
    sys.argv = [original_argv[0]] + remaining
    try:
        opt = TestOptions().parse(save=False)
    finally:
        sys.argv = original_argv

    phases = phase_args.phases
    if isinstance(phases, str):
        phase_list = [p.strip() for p in phases.split(",") if p.strip()]
    else:
        phase_list = ["test"]
    for phase in phase_list:
        _compute_psnr_for_phase(opt, phase)


if __name__ == "__main__":
    main()