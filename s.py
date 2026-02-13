import argparse
from PIL import Image
import numpy as np
from pathlib import Path
import cv2


def cleanup_corner_white_pixels(
    image,
    patch_ratio_h=0.12,
    patch_ratio_w=0.22,
    min_patch_h=36,
    min_patch_w=120,
    left_patch_length_scale=0.55,
    right_patch_length_scale=0.65,
    right_patch_height_scale=0.65,
):
    """
    Completely remove watermark regions by blacking out bottom-left and bottom-right
    corner patches.
    """
    width, height = image.size

    patch_h = min(height, max(min_patch_h, int(height * patch_ratio_h)))
    base_patch_w = min(width, max(min_patch_w, int(width * patch_ratio_w)))
    left_patch_w = max(1, int(base_patch_w * left_patch_length_scale))
    right_patch_w = max(1, int(base_patch_w * right_patch_length_scale))
    right_patch_h = max(1, int(patch_h * right_patch_height_scale))

    if image.mode == 'L':
        out = np.array(image)
        out[height - patch_h:, :left_patch_w] = 0
        out[height - right_patch_h:, width - right_patch_w:] = 0
        replaced_pixels = int((patch_h * left_patch_w) + (right_patch_h * right_patch_w))
        return Image.fromarray(out, mode='L'), replaced_pixels

    if image.mode == 'RGBA':
        out = np.array(image)
        out[height - patch_h:, :left_patch_w, 0] = 0
        out[height - patch_h:, :left_patch_w, 1] = 0
        out[height - patch_h:, :left_patch_w, 2] = 0
        out[height - right_patch_h:, width - right_patch_w:, 0] = 0
        out[height - right_patch_h:, width - right_patch_w:, 1] = 0
        out[height - right_patch_h:, width - right_patch_w:, 2] = 0
        replaced_pixels = int((patch_h * left_patch_w) + (right_patch_h * right_patch_w))
        return Image.fromarray(out, mode='RGBA'), replaced_pixels

    out = np.array(image.convert('RGB'))
    out[height - patch_h:, :left_patch_w] = [0, 0, 0]
    out[height - right_patch_h:, width - right_patch_w:] = [0, 0, 0]
    replaced_pixels = int((patch_h * left_patch_w) + (right_patch_h * right_patch_w))
    return Image.fromarray(out, mode='RGB'), replaced_pixels

def calculate_black_border_widths(image_path, threshold=10):
    """
    Calculate the width of black pixels at top, bottom, left, and right edges.
    
    Parameters:
    - image_path: Path to the image file
    - threshold: Pixel value threshold (0-255). Pixels below this are considered black.
                 Default is 10 to account for near-black pixels.
    
    Returns:
    - Dictionary with 'top', 'bottom', 'left', 'right' border widths
    """
    img = Image.open(image_path)
    img_array = np.array(img.convert('L'))
    
    height, width = img_array.shape
    
    # Calculate top border
    top = 0
    for i in range(height):
        if np.all(img_array[i, :] <= threshold):
            top += 1
        else:
            break
    
    # Calculate bottom border
    bottom = 0
    for i in range(height - 1, -1, -1):
        if np.all(img_array[i, :] <= threshold):
            bottom += 1
        else:
            break
    
    # Calculate left border
    left = 0
    for j in range(width):
        if np.all(img_array[:, j] <= threshold):
            left += 1
        else:
            break
    
    # Calculate right border
    right = 0
    for j in range(width - 1, -1, -1):
        if np.all(img_array[:, j] <= threshold):
            right += 1
        else:
            break
    
    return {
        'top': top,
        'bottom': bottom,
        'left': left,
        'right': right
    }

def crop_black_borders(image_path, output_path, threshold=10, corner_cleanup=False):
    """
    Crop black borders from all 4 edges and save the cropped image.
    Returns a dict with original size, cropped size, and detected border widths.
    """
    borders = calculate_black_border_widths(image_path, threshold=threshold)

    with Image.open(image_path) as img:
        width, height = img.size

        left = borders['left']
        top = borders['top']
        right = width - borders['right']
        bottom = height - borders['bottom']

        if left >= right or top >= bottom:
            cropped = img.copy()
        else:
            cropped = img.crop((left, top, right, bottom))

        corrected_pixels = 0
        if corner_cleanup:
            cropped, corrected_pixels = cleanup_corner_white_pixels(
                cropped,
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        cropped.save(output_path)

        return {
            'original_size': (width, height),
            'cropped_size': cropped.size,
            'borders': borders,
            'corner_white_to_black_pixels': corrected_pixels,
        }


def process_folder(input_folder, output_folder, threshold=10, corner_cleanup=False):
    files = sorted([p for p in input_folder.iterdir() if p.is_file()])

    print(f"\n=== Processing {input_folder.name} ===")
    print(f"Input files: {len(files)}")
    print(f"Output folder: {output_folder}")

    if not files:
        print("[WARNING] No files found.")
        return

    total_removed_w = 0
    total_removed_h = 0
    total_corrected_pixels = 0

    for path in files:
        output_path = output_folder / path.name
        result = crop_black_borders(
            path,
            output_path,
            threshold=threshold,
            corner_cleanup=corner_cleanup,
        )

        ow, oh = result['original_size']
        cw, ch = result['cropped_size']
        removed_w = ow - cw
        removed_h = oh - ch

        total_removed_w += removed_w
        total_removed_h += removed_h
        total_corrected_pixels += result['corner_white_to_black_pixels']

        print(
            f"{path.name} | borders:{result['borders']} | "
            f"size:{ow}x{oh} -> {cw}x{ch} | "
            f"corner_white_to_black:{result['corner_white_to_black_pixels']}"
        )

    n = len(files)
    print("\n-- Folder summary --")
    print(f"Processed: {n}")
    print(f"Avg removed width pixels: {total_removed_w / n:.2f}")
    print(f"Avg removed height pixels: {total_removed_h / n:.2f}")
    if corner_cleanup:
        print(f"Total corner white->black pixels: {total_corrected_pixels}")


def process_eye_dataset(dataset_dir, output_dir, threshold=10):
    dataset_path = Path(dataset_dir)
    output_path = Path(output_dir)

    subfolders = ['train_A', 'train_B', 'test_A', 'test_B']

    for folder_name in subfolders:
        input_folder = dataset_path / folder_name
        output_folder = output_path / folder_name

        if not input_folder.exists():
            print(f"[ERROR] Missing folder: {input_folder}")
            continue

        apply_corner_cleanup = folder_name in ['train_B', 'test_B']
        process_folder(
            input_folder,
            output_folder,
            threshold=threshold,
            corner_cleanup=apply_corner_cleanup,
        )


def _load_rgb_uint8(path):
    with Image.open(path) as img:
        return np.array(img.convert('RGB'), dtype=np.uint8)


def _save_rgb_uint8(path, arr):
    Image.fromarray(arr.astype(np.uint8), mode='RGB').save(path)


def _affine_register_rgb(moving_rgb, fixed_rgb, number_of_iterations=300, termination_eps=1e-6):
    fixed_gray = cv2.cvtColor(fixed_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    moving_gray = cv2.cvtColor(moving_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        int(number_of_iterations),
        float(termination_eps),
    )

    try:
        cv2.findTransformECC(
            fixed_gray,
            moving_gray,
            warp_matrix,
            cv2.MOTION_AFFINE,
            criteria,
        )
        height, width = fixed_rgb.shape[:2]
        registered = cv2.warpAffine(
            moving_rgb,
            warp_matrix,
            (width, height),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REFLECT,
        )
        return registered
    except cv2.error:
        return moving_rgb


def _match_color_lab(source_rgb, reference_rgb):
    source_lab = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    reference_lab = cv2.cvtColor(reference_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    src_mean = source_lab.reshape(-1, 3).mean(axis=0)
    src_std = source_lab.reshape(-1, 3).std(axis=0)
    ref_mean = reference_lab.reshape(-1, 3).mean(axis=0)
    ref_std = reference_lab.reshape(-1, 3).std(axis=0)

    adjusted = (source_lab - src_mean) * (ref_std / (src_std + 1e-6)) + ref_mean
    adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
    return cv2.cvtColor(adjusted, cv2.COLOR_LAB2RGB)


def apply_pairwise_postprocessing(
    output_dir,
    enable_affine_registration=False,
    enable_color_normalization=False,
    ecc_iterations=300,
    ecc_termination_eps=1e-6,
):
    if not enable_affine_registration and not enable_color_normalization:
        return

    def _process_folder_unpaired(folder_path):
        files = sorted([p for p in folder_path.iterdir() if p.is_file()])
        if not files:
            return 0

        reference_path = files[0]
        reference_rgb = _load_rgb_uint8(reference_path)
        processed = 0

        for image_path in files:
            image_rgb = _load_rgb_uint8(image_path)

            if enable_affine_registration and image_path != reference_path:
                image_rgb = _affine_register_rgb(
                    image_rgb,
                    reference_rgb,
                    number_of_iterations=ecc_iterations,
                    termination_eps=ecc_termination_eps,
                )

            if enable_color_normalization:
                image_rgb = _match_color_lab(image_rgb, reference_rgb)

            _save_rgb_uint8(image_path, image_rgb)
            processed += 1

        return processed

    output_path = Path(output_dir)
    for split in ['train', 'test']:
        split_A = output_path / f'{split}_A'
        split_B = output_path / f'{split}_B'

        if not split_A.exists() or not split_B.exists():
            continue

        a_files = {p.name: p for p in split_A.iterdir() if p.is_file()}
        b_files = {p.name: p for p in split_B.iterdir() if p.is_file()}
        common_names = sorted(set(a_files.keys()) & set(b_files.keys()))

        print(f"\n=== Postprocess {split} (matched pairs: {len(common_names)}) ===")

        if not common_names:
            print("No filename-matched A/B pairs found; using unpaired folder-wise reference mode.")
            processed_a = _process_folder_unpaired(split_A)
            processed_b = _process_folder_unpaired(split_B)
            print(f"Processed unpaired images: {split}_A={processed_a}, {split}_B={processed_b}")
            continue

        for name in common_names:
            a_path = a_files[name]
            b_path = b_files[name]

            fixed_a = _load_rgb_uint8(a_path)
            moving_b = _load_rgb_uint8(b_path)

            if enable_affine_registration:
                moving_b = _affine_register_rgb(
                    moving_b,
                    fixed_a,
                    number_of_iterations=ecc_iterations,
                    termination_eps=ecc_termination_eps,
                )

            if enable_color_normalization:
                moving_b = _match_color_lab(moving_b, fixed_a)

            _save_rgb_uint8(b_path, moving_b)


def _build_arg_parser():
    parser = argparse.ArgumentParser()
    base_dir = Path(__file__).resolve().parent
    parser.add_argument('--dataset_dir', type=Path, default=base_dir / 'datasets' / 'eye')
    parser.add_argument('--output_dir', type=Path, default=base_dir / 'datasets' / 'eye_cropped')
    parser.add_argument('--threshold', type=int, default=10)
    parser.add_argument('--enable_affine_registration', action='store_true')
    parser.add_argument('--enable_color_normalization', action='store_true')
    parser.add_argument('--ecc_iterations', type=int, default=300)
    parser.add_argument('--ecc_termination_eps', type=float, default=1e-6)
    return parser


if __name__ == '__main__':
    args = _build_arg_parser().parse_args()
    process_eye_dataset(args.dataset_dir, args.output_dir, threshold=args.threshold)
    apply_pairwise_postprocessing(
        args.output_dir,
        enable_affine_registration=args.enable_affine_registration,
        enable_color_normalization=args.enable_color_normalization,
        ecc_iterations=args.ecc_iterations,
        ecc_termination_eps=args.ecc_termination_eps,
    )
