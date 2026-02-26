import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, random_split
from transformers import (
    SegformerImageProcessor, 
    SegformerForSemanticSegmentation, 
    Trainer, 
    TrainingArguments
)
import evaluate

# 1. Custom PyTorch Dataset
class GingivaDataset(Dataset):
    def __init__(self, images_dir, masks_dir, image_processor):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_processor = image_processor
        
        valid_extensions = ('.png', '.jpg', '.jpeg')
        self.images = [f for f in sorted(os.listdir(images_dir)) if f.lower().endswith(valid_extensions)]
        
        self.masks = [os.path.splitext(img)[0] + "_mask.png" for img in self.images]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.masks[idx])
        
        image = Image.open(img_path).convert("RGB")
        
        # If the mask doesn't exist (e.g., no annotation), create an empty one
        if os.path.exists(mask_path):
            mask = np.array(Image.open(mask_path).convert("L"))
            # Mask from xml_to_mask is 255 for gingiva. We need 0 (bg) and 1 (gingiva)
            mask = (mask > 0).astype(np.uint8) 
        else:
            width, height = image.size
            mask = np.zeros((height, width), dtype=np.uint8)
            
        # The image processor normalizes the image, resizes it, and returns PyTorch tensors
        encoded_inputs = self.image_processor(image, mask, return_tensors="pt")
        
        # Remove the batch dimension added by the processor
        for k, v in encoded_inputs.items():
            encoded_inputs[k] = v.squeeze() 

        return encoded_inputs


# 2. Main Training Function
def main():
    # Adjust paths to where your dataset and masks folders are located
    # Ensure no nested folders inside dataset or masks directories
    DATASET_DIR = "image_test" 
    MASKS_DIR = "masks"
    
    PRETRAINED_MODEL = "nvidia/segformer-b2-finetuned-ade-512-512"
    
    print("Loading Image Processor...")
    # reduce_labels=False since our background is 0 and object is 1
    image_processor = SegformerImageProcessor.from_pretrained(PRETRAINED_MODEL, do_reduce_labels=False)
    
    print("Loading Dataset...")
    full_dataset = GingivaDataset(DATASET_DIR, MASKS_DIR, image_processor)
    
    if len(full_dataset) == 0:
        raise ValueError(f"No images found in {DATASET_DIR}. Please check the path.")
    
    # Split Dataset into Train/Validation/Test (70/15/15)
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size # Remainder goes to test
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    print(f"Dataset split: {train_size} training, {val_size} validation, {test_size} test images.")

    # --- Print the names of the test images ---
    print("\n--- Test Set Images ---")
    test_image_names = [full_dataset.images[i] for i in test_dataset.indices]
    for name in test_image_names:
        print(name)
    print("-----------------------\n")

    # Label mappings
    id2label = {0: "background", 1: "keratinized_gingiva"}
    label2id = {"background": 0, "keratinized_gingiva": 1}
    
    print("Initializing Model...")
    model = SegformerForSemanticSegmentation.from_pretrained(
        PRETRAINED_MODEL,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True # Required because we are changing the number of classes from 150 to 2
    )
    
    # Define Metrics using Hugging Face evaluate
    metric = evaluate.load("mean_iou")
    
    def compute_metrics(eval_pred):
        with torch.no_grad():
            logits, labels = eval_pred
            logits_tensor = torch.from_numpy(logits)
            
            # Segformer outputs logits 1/4th the size of the images. Need to upsample.
            logits_tensor = torch.nn.functional.interpolate(
                logits_tensor,
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).argmax(dim=1)
            
            pred_labels = logits_tensor.detach().cpu().numpy()
            
            # Compute mIoU
            metrics = metric.compute(
                predictions=pred_labels,
                references=labels,
                num_labels=2,
                ignore_index=255, # By default HF ignores 255
                reduce_labels=False,
            )
            
            # Convert numpy arrays to lists for Trainer compatibility
            for key, value in metrics.items():
                if isinstance(value, np.ndarray):
                    metrics[key] = value.tolist()
            return metrics
    
    # Training Arguments
    training_args = TrainingArguments(
        output_dir="./segformer-gingiva-output",
        use_cpu=True,
        learning_rate=6e-5,
        num_train_epochs=50, 
        per_device_train_batch_size=4, 
        per_device_eval_batch_size=4,
        save_total_limit=3,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_steps=20,
        eval_steps=20,
        logging_steps=10,
        eval_accumulation_steps=5, # Offload to CPU to avoid OOM during evaluation
        remove_unused_columns=False, # Important: Segformer expects specific dictionary keys
        push_to_hub=False,
        load_best_model_at_end=True,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    print("Starting Training...")
    trainer.train()
    
    print("Training Complete! Saving final model...")
    trainer.save_model("./segformer-gingiva-final")
    image_processor.save_pretrained("./segformer-gingiva-final")

if __name__ == "__main__":
    main()
