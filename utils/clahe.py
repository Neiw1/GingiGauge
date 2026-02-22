import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Load image ---
image_path = "Dataset/Test/Images/00933.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# --- Convert to LAB color space ---
lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
l, a, b = cv2.split(lab)

# --- Apply CLAHE on L-channel (lightness only) ---
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
l_clahe = clahe.apply(l)

# --- Merge back and convert to RGB ---
lab_clahe = cv2.merge((l_clahe, a, b))
image_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

# --- Display before & after ---
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(image_clahe)
plt.title("After CLAHE Enhancement")
plt.axis("off")

plt.tight_layout()
plt.show()

cv2.imwrite('clahe_output.jpg', image_clahe[:, :, ::-1])  # Save as BGR format