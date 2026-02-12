import os
import shutil
from PIL import Image

def process_images():
    source_dir = 'datasets/eye'
    dest_dir = 'datasets/eye_cut'
    
    # Create destination directory if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    subfolders = ['train_A', 'train_B', 'test_A', 'test_B']
    
    for subfolder in subfolders:
        src_subfolder = os.path.join(source_dir, subfolder)
        dest_subfolder = os.path.join(dest_dir, subfolder)
        
        # Create subfolder in dest
        if not os.path.exists(dest_subfolder):
            os.makedirs(dest_subfolder)
        
        if not os.path.exists(src_subfolder):
            print(f"Source subfolder {src_subfolder} does not exist. Skipping.")
            continue
        
        for filename in os.listdir(src_subfolder):
            src_path = os.path.join(src_subfolder, filename)
            dest_path = os.path.join(dest_subfolder, filename)
            
            if not os.path.isfile(src_path):
                continue
            
            # Check if image (basic check)
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                # Copy non-image files as is
                shutil.copy2(src_path, dest_path)
                continue
            
            # Load image
            try:
                img = Image.open(src_path)
                width, height = img.size
                
                # Check if filename ends with OD or OS (before extension)
                name_parts = filename.rsplit('.', 1)
                if len(name_parts) == 2:
                    base_name, ext = name_parts
                    if base_name.endswith('OD'):
                        # Keep right half
                        cropped = img.crop((width // 2, 0, width, height))
                        cropped.save(dest_path)
                        print(f"Processed OD: {filename}")
                    elif base_name.endswith('OS'):
                        # Keep left half
                        cropped = img.crop((0, 0, width // 2, height))
                        cropped.save(dest_path)
                        print(f"Processed OS: {filename}")
                    else:
                        # Copy as is
                        shutil.copy2(src_path, dest_path)
                        print(f"Copied: {filename}")
                else:
                    # No extension or weird, copy
                    shutil.copy2(src_path, dest_path)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                # Copy anyway
                shutil.copy2(src_path, dest_path)

if __name__ == "__main__":
    process_images()