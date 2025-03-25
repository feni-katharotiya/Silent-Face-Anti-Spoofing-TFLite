import os
import glob
import cv2

from tqdm import tqdm

def resize_to_aspect(image_path, output_path, aspect_ratio=(4, 3)):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return
    img_height, img_width = img.shape[:2]
    target_aspect = aspect_ratio[0] / aspect_ratio[1]
    
    # Determine current aspect ratio
    current_aspect = img_width / img_height

    if current_aspect > target_aspect:
        # Image is wider than the target aspect ratio, need to crop width
        new_width = int(target_aspect * img_height)
        left = (img_width - new_width) // 2
        right = left + new_width
        img_cropped = img[:, left:right]
    else:
        # Image is taller than the target aspect ratio, need to crop height
        new_height = int(img_width / target_aspect)
        top = (img_height - new_height) // 2
        bottom = top + new_height
        img_cropped = img[top:bottom, :]

    # Resize to specific size
    resized_img = cv2.resize(img_cropped, (aspect_ratio[0] * 100, aspect_ratio[1] * 100), interpolation=cv2.INTER_AREA)

    cv2.imwrite(output_path, resized_img)

def process_images(input_folder, aspect_ratio=(4, 3)):
    # Create output folder path
    output_folder = f"{input_folder}_aspect_resized"
    os.makedirs(output_folder, exist_ok=True)

    # Read all image files from the input folder
    image_files = glob.glob(os.path.join(input_folder, '*'))
    
    for image_file in tqdm(image_files):
        # Get the base name of the file
        base_name = os.path.basename(image_file)
        
        # Create output file path
        output_file = os.path.join(output_folder, base_name)
        
        # Resize the image
        resize_to_aspect(image_file, output_file, aspect_ratio)
    
    print(f"Images saved to: {output_folder}")

# Example usage
input_folder = "/home/deepkaneria/HDD/Shared/Datasets/Face/Anit-Spoofing/MobileCaptures/full/Spoof"
process_images(input_folder, (3, 4))
