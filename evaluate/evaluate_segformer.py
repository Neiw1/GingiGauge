import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from pathlib import Path

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
    # Setup paths
    MODEL_DIR = "segformer-gingiva-final"
    TEST_DIR = Path("test_data")
    MASKS_DIR = Path("masks_test")
    OUTPUT_DIR = Path("prediction_results")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if not os.path.exists(MODEL_DIR):
        print(f"Error: Model directory '{MODEL_DIR}' not found.")
        return
        
    print("Loading Model and Processor...")
    image_processor = SegformerImageProcessor.from_pretrained(MODEL_DIR)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_DIR)
    
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}")
    model.to(device)
    model.eval()
    
    iou_scores = []
    dsc_scores = []
    
    # Process all images in test_data
    image_paths = list(TEST_DIR.glob("*.*"))
    if not image_paths:
        print(f"No images found in {TEST_DIR}")
        return
        
    print(f"Found {len(image_paths)} images to process.\n")
    
    for img_path in image_paths:
        # Ignore non-images like .DS_Store
        if img_path.name.startswith('.'):
            continue
            
        base_name = img_path.stem
        # Corresponding mask should be named exactly like this based on our generation script
        true_mask_path = MASKS_DIR / f"{base_name}_mask.png"
        
        # 1. Load Original Image
        image = Image.open(img_path).convert("RGB")
        
        # 2. Run Inference
        inputs = image_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            
        logits = outputs.logits
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=image.size[::-1], 
            mode="bilinear",
            align_corners=False
        )
        predicted_mask = upsampled_logits.argmax(dim=1).squeeze().cpu().numpy()
        
        # 3. Load Ground Truth Mask (if it exists)
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
            # Create a blank array so visual comparison doesn't break
            true_mask = np.zeros_like(predicted_mask)
            
        print(f"Processed: {base_name:<10} | IoU: {img_iou:.4f} | DSC: {img_dsc:.4f}")
        
        # 4. Generate Visual Comparisons
        pred_overlay = overlay_mask(image, predicted_mask, color=(255, 0, 0)) # Red for prediction
        true_overlay = overlay_mask(image, true_mask, color=(0, 255, 0))      # Green for ground truth
        
        # 5. Create a side-by-side plot
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
