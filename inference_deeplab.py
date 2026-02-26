import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

# Important: Use Albumentations so the inference preprocessing exactly matches training
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

def compute_metrics(pred_mask, true_mask):
    """Calculates Intersection over Union (IoU) and Dice Similarity Coefficient (DSC)."""
    pred_mask = (pred_mask > 0).astype(np.uint8)
    true_mask = (true_mask > 0).astype(np.uint8)
    
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    
    iou = (intersection + 1e-6) / (union + 1e-6)
    
    pred_sum = pred_mask.sum()
    true_sum = true_mask.sum()
    dsc = (2.0 * intersection + 1e-6) / (pred_sum + true_sum + 1e-6)
    
    return iou, dsc

def overlay_mask(image, mask, alpha=0.5, color=(0, 255, 0)):
    # If image is a string path, load it; if PIL Image, handle correctly
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    
    img_cv = np.array(image)
    if not isinstance(image, Image.Image): # Assuming it's already an array
        img_cv = np.array(image)
    elif isinstance(image, Image.Image):
        img_cv = np.array(image.convert("RGB"))
        
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    
    colored_mask = np.zeros_like(img_cv)
    colored_mask[mask == 1] = color 
    
    overlay = cv2.addWeighted(img_cv, 1 - alpha, colored_mask, alpha, 0)
    result = np.where(colored_mask == 0, img_cv, overlay)
    
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

def main():
    MODEL_PATH = "best_deeplabv3plus.pth"
    TEST_DIR = Path("test_data")
    MASKS_DIR = Path("masks_test")
    OUTPUT_DIR = Path("prediction_results_deeplab")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model weights '{MODEL_PATH}' not found. Did you download it from Kaggle?")
        return
        
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}")
    
    print("Initializing Model...")
    
    # 1. Initialize the EXACT SAME model architecture from segmentation_models_pytorch
    model = smp.DeepLabV3Plus(
        encoder_name="resnet50",        
        encoder_weights=None, # We load weights manually next    
        in_channels=3,                  
        classes=1,            # 1 class for binary segmentation (Gingiva vs Background)                      
    )
    
    # Load weights
    try:
        state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading state dict: {e}")
        return
        
    model.to(device)
    model.eval()
    
    # 2. Match the inference transformations directly from training code
    transform = A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    iou_scores = []
    dsc_scores = []
    
    image_paths = list(TEST_DIR.glob("*.*"))
    if not image_paths:
        print(f"No images found in {TEST_DIR}")
        return
        
    print(f"Found {len(image_paths)} images to process.\n")
    
    for img_path in image_paths:
        if img_path.name.startswith('.'):
            continue
            
        base_name = img_path.stem
        true_mask_path = MASKS_DIR / f"{base_name}_mask.png"
        
        # Load Original Image as Numpy Array for transform
        image_pil = Image.open(img_path).convert("RGB")
        image_np = np.array(image_pil)
        orig_h, orig_w = image_np.shape[:2]
        
        # Transform
        augmented = transform(image=image_np)
        input_tensor = augmented['image'].unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = model(input_tensor) # Shape: (1, 1, 512, 512)
            
        # Resize logits back to original image size
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=(orig_h, orig_w), 
            mode="bilinear",
            align_corners=False
        )
        
        # Apply Sigmoid and 0.5 Threshold for Binary Segmentation!
        probs = torch.sigmoid(upsampled_logits).squeeze().cpu().numpy()
        predicted_mask = (probs > 0.5).astype(np.uint8)
        
        img_iou, img_dsc = 0.0, 0.0
        
        if true_mask_path.exists():
            true_mask_img = Image.open(true_mask_path).convert("L")
            true_mask = np.array(true_mask_img)
            true_mask = (true_mask > 0).astype(np.uint8)
            
            img_iou, img_dsc = compute_metrics(predicted_mask, true_mask)
            iou_scores.append(img_iou)
            dsc_scores.append(img_dsc)
        else:
            print(f"Warning: Missing ground truth for {base_name}.")
            true_mask = np.zeros_like(predicted_mask)
            
        print(f"Processed: {base_name:<10} | IoU: {img_iou:.4f} | DSC: {img_dsc:.4f}")
        
        pred_overlay = overlay_mask(image_pil, predicted_mask, color=(255, 0, 0)) # Red
        true_overlay = overlay_mask(image_pil, true_mask, color=(0, 255, 0))      # Green
        
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f"Image: {img_path.name}\nDeepLabV3+ | IoU: {img_iou:.4f} | DSC: {img_dsc:.4f}", fontsize=16)
        
        axs[0, 0].imshow(image_np)
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
        out_file = OUTPUT_DIR / f"{base_name}_deeplab_comparison.jpg"
        plt.savefig(out_file, dpi=150, bbox_inches='tight')
        plt.close()

    if iou_scores:
        avg_iou = np.mean(iou_scores)
        avg_dsc = np.mean(dsc_scores)
        print("\n" + "="*40)
        print(f"DEEPLAB FINAL METRICS ON {len(iou_scores)} IMAGES:")
        print(f"Average IoU: {avg_iou:.4f}")
        print(f"Average DSC: {avg_dsc:.4f}")
        print(f"Results saved to: {OUTPUT_DIR.absolute()}")
        print("="*40)

if __name__ == "__main__":
    main()
