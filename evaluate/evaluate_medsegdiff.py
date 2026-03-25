import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import MedSegDiff architecture
from med_seg_diff_pytorch import Unet, MedSegDiff

def compute_metrics(pred_mask, true_mask):
    """Calculates Intersection over Union (IoU) and Dice Similarity Coefficient (DSC)."""
    # Ensure binary arrays
    pred_mask = (pred_mask > 0).astype(np.uint8)
    true_mask = (true_mask > 0).astype(np.uint8)
    
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    
    # Avoid division by zero
    iou = (intersection + 1e-6) / (union + 1e-6)
    
    # DSC = 2 * Intersection / (Pred + True)
    pred_sum = pred_mask.sum()
    true_sum = true_mask.sum()
    dsc = (2.0 * intersection + 1e-6) / (pred_sum + true_sum + 1e-6)
    
    return iou, dsc

def overlay_mask(image, mask, alpha=0.5, color=(0, 255, 0)):
    """ Overlays the binary segmentation mask onto the target image. """
    img_cv = np.array(image.convert("RGB"))
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    
    colored_mask = np.zeros_like(img_cv)
    colored_mask[mask == 1] = color 
    
    overlay = cv2.addWeighted(img_cv, 1 - alpha, colored_mask, alpha, 0)
    result = np.where(colored_mask == 0, img_cv, overlay)
    
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

def main():
    # Setup paths - Change these to your specific Kaggle directories if needed
    MODEL_PATH = "best_medsegdiff.pth" # Points directly to the saved weights file
    TEST_DIR = Path("test_data")
    MASKS_DIR = Path("masks_test")
    OUTPUT_DIR = Path("prediction_results_medsegdiff")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found.")
        return
        
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}")
    
    print("Initializing MedSegDiff Architecture...")
    # 1. Recreate the exact architecture used in training
    unet = Unet(
        dim = 64,
        image_size = 256,
        mask_channels = 1,
        input_img_channels = 3,
        dim_mults = (1, 2, 4, 8)
    )
    model = MedSegDiff(
        unet,
        timesteps = 1000
    )
    
    print("Loading Weights...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    
    # 2. Setup the exact Image Preprocessor used in training
    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255.0),
        ToTensorV2(),
    ])
    
    iou_scores = []
    dsc_scores = []
    
    # Process all images in test_data
    image_paths = list(TEST_DIR.glob("*.*"))
    if not image_paths:
        print(f"No images found in {TEST_DIR}")
        return
        
    print(f"Found {len(image_paths)} images to process.\n")
    print("Note: Diffusion inference takes 1000 steps per image. This will take a few minutes...")
    
    for img_path in image_paths:
        if img_path.name.startswith('.'):
            continue
            
        base_name = img_path.stem
        true_mask_path = MASKS_DIR / f"{base_name}_mask.png"
        
        # 1. Load Original Image
        image = Image.open(img_path).convert("RGB")
        original_size = image.size # (width, height)
        image_np = np.array(image)
        
        # 2. Preprocess
        augmented = transform(image=image_np)
        image_tensor = augmented['image'].unsqueeze(0).to(device)
        
        # 3. Run Inference (Diffusion Denoising)
        with torch.no_grad():
            pred_mask_tensor = model.sample(image_tensor)
            
        # 4. Post-process (Upsample and Threshold)
        upsampled_mask = torch.nn.functional.interpolate(
            pred_mask_tensor,
            size=(original_size[1], original_size[0]), # (height, width)
            mode="bilinear",
            align_corners=False
        )
        
        # Diffusion outputs continuous values, so we threshold at 0.5 to make it binary
        predicted_mask = (upsampled_mask.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
        
        # 5. Load Ground Truth Mask (if it exists)
        img_iou = 0.0
        img_dsc = 0.0
        
        if true_mask_path.exists():
            true_mask_img = Image.open(true_mask_path).convert("L")
            true_mask = np.array(true_mask_img)
            # Binary mapping: anything > 0 is gingiva (1)
            true_mask = (true_mask > 0).astype(np.uint8)
            
            img_iou, img_dsc = compute_metrics(predicted_mask, true_mask)
            iou_scores.append(img_iou)
            dsc_scores.append(img_dsc)
        else:
            print(f"Warning: Missing ground truth for {base_name}. Skipping metrics.")
            true_mask = np.zeros_like(predicted_mask)
            
        print(f"Processed: {base_name:<10} | IoU: {img_iou:.4f} | DSC: {img_dsc:.4f}")
        
        # 6. Generate Visual Comparisons
        pred_overlay = overlay_mask(image, predicted_mask, color=(255, 0, 0)) # Red for prediction
        true_overlay = overlay_mask(image, true_mask, color=(0, 255, 0))      # Green for ground truth
        
        # 7. Create a side-by-side plot
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f"Image: {img_path.name}\nIoU: {img_iou:.4f} | DSC: {img_dsc:.4f}", fontsize=16)
        
        axs[0, 0].imshow(image)
        axs[0, 0].set_title("Original Image")
        axs[0, 0].axis("off")
        
        axs[0, 1].imshow(true_mask, cmap="gray")
        axs[0, 1].set_title("Ground Truth Mask")
        axs[0, 1].axis("off")
        
        axs[1, 0].imshow(true_overlay)
        axs[1, 0].set_title("Ground Truth Overlay (Green)")
        axs[1, 0].axis("off")
        
        axs[1, 1].imshow(pred_overlay)
        axs[1, 1].set_title("Prediction Overlay (Red)")
        axs[1, 1].axis("off")
        
        plt.tight_layout()
        out_file = OUTPUT_DIR / f"{base_name}_comparison.jpg"
        plt.savefig(out_file, dpi=150, bbox_inches='tight')
        plt.close()

    # Final Summary
    if iou_scores:
        avg_iou = np.mean(iou_scores)
        avg_dsc = np.mean(dsc_scores)
        print("\n" + "="*40)
        print(f"FINAL METRICS ON {len(iou_scores)} IMAGES:")
        print(f"Average IoU: {avg_iou:.4f}")
        print(f"Average DSC: {avg_dsc:.4f}")
        print(f"Results saved to: {OUTPUT_DIR.absolute()}")
        print("="*40)

if __name__ == "__main__":
    main()