import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

def overlay_mask(image, mask, alpha=0.5, color=(0, 255, 0)):
    """ Overlays the binary segmentation mask onto the target image. """
    # Convert PIL Image to cv2 format (numpy array)
    img_cv = np.array(image.convert("RGB"))
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    
    # Create colored mask
    colored_mask = np.zeros_like(img_cv)
    colored_mask[mask == 1] = color # Apply color to gingiva pixels
    
    # Blend the image and mask
    overlay = cv2.addWeighted(img_cv, 1 - alpha, colored_mask, alpha, 0)
    
    # Only show the mask over the specific region
    result = np.where(colored_mask == 0, img_cv, overlay)
    
    # Convert back to RGB for matplotlib/PIL
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

def main():
    # Setup paths
    MODEL_DIR = "./segformer-gingiva-final"
    TEST_IMAGE_PATH = "image_test/00733.jpg"
    
    if not os.path.exists(MODEL_DIR):
        print(f"Error: Model directory '{MODEL_DIR}' not found. Did you run the training script?")
        return
        
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"Error: Test image '{TEST_IMAGE_PATH}' not found. Please provide a valid path.")
        
        # If user is just looking for the file creation, we don't need to crash.
        return

    print("Loading Model and Processor...")
    image_processor = SegformerImageProcessor.from_pretrained(MODEL_DIR)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_DIR)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    print(f"Processing image: {TEST_IMAGE_PATH}")
    image = Image.open(TEST_IMAGE_PATH)
    
    # Prepare image for model
    inputs = image_processor(images=image, return_tensors="pt").to(device)
    
    # Inference
    print("Running inference...")
    with torch.no_grad():
        outputs = model(**inputs)
        
    # Resize outputs (logits) to match the original image size
    logits = outputs.logits  # shape (batch_size, num_classes, height/4, width/4)
    
    # Interpolate to original size
    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=image.size[::-1], # (height, width)
        mode="bilinear",
        align_corners=False
    )
    
    # Get the class predictions (Background=0, Gingiva=1)
    predicted_mask = upsampled_logits.argmax(dim=1).squeeze().cpu().numpy()
    
    # Generate Overlay
    print("Generating overlay...")
    result_image = overlay_mask(image, predicted_mask)
    
    # Save Strategy
    output_filename = "inference_output_large.jpg"
    plt.imsave(output_filename, result_image)
    print(f"Saved inference result to {output_filename}")
    
    # Display the result side-by-side
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(image)
    axs[0].set_title("Original Image")
    axs[0].axis("off")
    
    axs[1].imshow(predicted_mask, cmap="gray")
    axs[1].set_title("Predicted Mask (Keratinized Gingiva)")
    axs[1].axis("off")
    
    axs[2].imshow(result_image)
    axs[2].set_title("Overlay")
    axs[2].axis("off")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
