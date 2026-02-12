from PIL import Image
import numpy as np
from pathlib import Path


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


if __name__ == '__main__':
    base_dir = Path(__file__).resolve().parent
    eye_dataset_dir = base_dir / 'datasets' / 'eye'
    eye_output_dir = base_dir / 'datasets' / 'eye_cropped'

    process_eye_dataset(eye_dataset_dir, eye_output_dir, threshold=10)
