import os
from PIL import Image

def get_image_resolutions(folder_path):
    resolutions = {}
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_path = os.path.join(folder_path, filename)
            try:
                with Image.open(image_path) as img:
                    resolutions[filename] = img.size  # (width, height)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    return resolutions

# Example usage
folder_path = 'datasets/eye_cropped/train_B'  # Replace with actual folder path
resolutions = get_image_resolutions(folder_path)
for filename, size in resolutions.items():
    print(f"{filename}: {size[0]}x{size[1]}")