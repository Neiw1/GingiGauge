import argparse
import os
import torch
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

def process_image(image_path, output_path, model, image_processor, device):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error reading {image_path}: {e}")
        return

    inputs = image_processor(images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    logits = outputs.logits
    # Resize logits back to original image size
    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=image.size[::-1], 
        mode="bilinear",
        align_corners=False
    )
    predicted_mask = upsampled_logits.argmax(dim=1).squeeze().cpu().numpy()
    
    # Save mask (255 for gingiva, 0 for background) for visibility
    mask_to_save = (predicted_mask > 0).astype(np.uint8) * 255
    cv2.imwrite(str(output_path), mask_to_save)
    print(f"Processed: {image_path.name:<20} -> {output_path.name}")

def main():
    parser = argparse.ArgumentParser(description="Predict mask for image(s) using SegFormer")
    parser.add_argument("--input", type=str, default="images/", help="Path to the input image or folder of images (default: images/)")
    parser.add_argument("--model_path", type=str, default="segformer-gingiva-final", help="Path to the trained SegFormer model directory")
    parser.add_argument("--output", type=str, default="results/", help="Path to save the predicted mask(s) (default: results/)")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input path '{input_path}' not found.")
        return

    if not os.path.exists(args.model_path):
        print(f"Error: Model directory '{args.model_path}' not found. Ensure it is in the current directory.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}")
    
    print("Loading SegFormer Model and Processor...")
    image_processor = SegformerImageProcessor.from_pretrained(args.model_path)
    model = SegformerForSemanticSegmentation.from_pretrained(args.model_path)
    
    model.to(device)
    model.eval()
    
    if input_path.is_file():
        # Input is a single file
        # Check if output is meant to be a directory or a specific file
        if output_path.suffix == '' or args.output.endswith('/') or args.output.endswith('\\'):
            # Treat output as directory
            output_path.mkdir(parents=True, exist_ok=True)
            out_file = output_path / f"{input_path.stem}_mask.png"
        else:
            # Treat output as exact file path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            out_file = output_path
            
        process_image(input_path, out_file, model, image_processor, device)
        
    elif input_path.is_dir():
        # Input is a directory
        output_path.mkdir(parents=True, exist_ok=True)
        valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        
        image_files = [f for f in input_path.glob('*.*') if f.suffix.lower() in valid_extensions and not f.name.startswith('.')]
        
        if not image_files:
            print(f"No valid images found in directory: {input_path}")
            return
            
        print(f"Found {len(image_files)} images in '{input_path}'. Saving to '{output_path}'...\n")
        
        for img_path in image_files:
            out_file = output_path / f"{img_path.stem}_mask.png"
            process_image(img_path, out_file, model, image_processor, device)
            
        print(f"\nSuccessfully finished processing {len(image_files)} images!")
    else:
        print("Invalid input path.")

if __name__ == "__main__":
    main()
