import xml.etree.ElementTree as ET
import numpy as np
import cv2
import os

# Create output folder
output_folder = 'masks_test'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

tree = ET.parse('annotations_val.xml')
root = tree.getroot()

# Use .//image to ensure we find all image nodes
images = root.findall(".//image")
print(f"Found {len(images)} total image entries.")

processed_count = 0

for image_node in images:
    img_id = image_node.attrib['id']
    img_name = image_node.attrib['name']
    width = int(image_node.attrib['width'])
    height = int(image_node.attrib['height'])

    polylines = image_node.findall("polyline")
    polygons = image_node.findall("polygon")
    masks = image_node.findall("mask")
    
    # Check if this image has ANY annotations
    if not (polylines or polygons or masks):
        continue

    # Create blank mask
    mask_img = np.zeros((height, width), dtype=np.uint8)

    # Process Polylines and Polygons
    for poly in polylines + polygons:
        points_str = poly.attrib['points']
        points = []
        for p in points_str.split(';'):
            if not p.strip(): continue
            x, y = map(float, p.split(','))
            points.append([x, y])
        
        if points:
            points_array = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask_img, [points_array], color=255)

    # Process RLE Masks
    for m in masks:
        # Expected format: <mask rle="1, 2, 3..." left="x" top="y" width="w" height="h" ... />
        if 'rle' in m.attrib:
            rle_str = m.attrib['rle']
            left = int(m.attrib.get('left', 0))
            top = int(m.attrib.get('top', 0))
            w = int(m.attrib.get('width', width))
            h = int(m.attrib.get('height', height))
            
            counts = [int(x.strip()) for x in rle_str.split(',')]
            
            # Reconstruct mask ROI
            mask_roi = np.zeros(w * h, dtype=np.uint8)
            idx = 0
            val = 0
            for count in counts:
                if idx + count > len(mask_roi):
                    # Clip if exceeds bounds
                    mask_roi[idx:] = val
                    break
                mask_roi[idx : idx+count] = val
                idx += count
                val = 1 - val
            
            # CVAT masks are typically Row-major
            mask_roi_2d = mask_roi.reshape((h, w)) * 255
            
            # Overlay onto main mask using logical OR 
            # (if mask goes out of bounds, we clip it safely)
            y1, y2 = max(0, top), min(height, top + h)
            x1, x2 = max(0, left), min(width, left + w)
            
            roi_y1, roi_y2 = y1 - top, (y2 - y1) + (y1 - top)
            roi_x1, roi_x2 = x1 - left, (x2 - x1) + (x1 - left)
            
            if y2 > y1 and x2 > x1:
                mask_img[y1:y2, x1:x2] = np.maximum(
                    mask_img[y1:y2, x1:x2], 
                    mask_roi_2d[roi_y1:roi_y2, roi_x1:roi_x2]
                )

    # Save logic
    clean_name = os.path.basename(img_name)
    base_name = os.path.splitext(clean_name)[0]
    output_path = os.path.join(output_folder, f"{base_name}_mask.png")
    
    cv2.imwrite(output_path, mask_img)
    processed_count += 1
    print(f"Generated: {output_path} (ID: {img_id})")

print(f"\nFinished! Processed {processed_count} annotated images.")