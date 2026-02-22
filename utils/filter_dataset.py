import os
import shutil
from pathlib import Path

def create_dataset_with_masks():
    # Define paths
    base_dir = Path('/Users/neiw/Desktop/Final')
    full_dataset_dir = base_dir / 'Dataset'  # the folder containing ALL images (assuming this is where they are)
    masks_dir = base_dir / 'masks_test'      # the folder with the generated masks
    output_dir = base_dir / 'dataset_with_mask'
    
    # Create the output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Checking for images in: {full_dataset_dir}")
    print(f"Based on masks in: {masks_dir}")
    
    if not masks_dir.exists():
        print(f"Error: Masks directory '{masks_dir}' does not exist.")
        return
        
    if not full_dataset_dir.exists():
        print(f"Error: Full dataset directory '{full_dataset_dir}' does not exist.")
        # If it's located somewhere else (e.g., image_test), please adjust the full_dataset_dir above
        return

    # Keep track of counts
    copied_count = 0
    missing_images = []
    
    # Find all masks
    for mask_path in masks_dir.glob('**/*_mask.png'):
        # Extract the base name (e.g., '00929_mask.png' -> '00929')
        base_name = mask_path.name.replace('_mask.png', '')
        
        # We don't know the exact extension of the original image (.jpg, .jpeg, .png), 
        # so we search for any file matching the base name
        matching_images = list(full_dataset_dir.rglob(f"{base_name}.*"))
        
        if not matching_images:
            # Maybe the Images/ prefix from XML affects things, try searching without the path
            matching_images = [img for img in full_dataset_dir.rglob('*') 
                               if img.is_file() and img.stem == base_name]
                               
        if matching_images:
            # We found a matching image! Let's take the first one (in case there are multiple types)
            source_img_path = matching_images[0]
            
            # Destination path will have the exact same file name as the original image
            dest_img_path = output_dir / source_img_path.name
            
            # Copy the file
            shutil.copy2(source_img_path, dest_img_path)
            copied_count += 1
            print(f"Copied: {source_img_path.name} -> {output_dir.name}/")
        else:
            missing_images.append(base_name)
            
    print("\n--- Summary ---")
    print(f"Total masks found: {len(list(masks_dir.glob('**/*_mask.png')))}")
    print(f"Successfully copied images: {copied_count}")
    
    if missing_images:
        print(f"\nCould not find original images for {len(missing_images)} masks:")
        # Print first 10 missing to avoid flooding console
        print(", ".join(missing_images[:10]) + ("..." if len(missing_images) > 10 else ""))

if __name__ == "__main__":
    create_dataset_with_masks()
