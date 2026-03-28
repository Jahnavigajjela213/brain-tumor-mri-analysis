import os
import nibabel as nib
import numpy as np
import cv2

# ✅ Correct path (your dataset location)
INPUT_PATH = r"C:\Users\girid\OneDrive\Desktop\brain-tumor-backend\BraTS2020_TrainingData"

# ✅ FIXED: correct relative output (since you're inside backend)
OUTPUT_IMG = "dataset/images"
OUTPUT_MASK = "dataset/masks"

os.makedirs(OUTPUT_IMG, exist_ok=True)
os.makedirs(OUTPUT_MASK, exist_ok=True)

count = 0
limit = 50

print(f"Starting BraTS processing from {INPUT_PATH}...")

if not os.path.exists(INPUT_PATH):
    print(f"Error: INPUT_PATH '{INPUT_PATH}' not found.")
    exit(1)

# 🔥 IMPORTANT: walk through ALL nested folders
for root, dirs, files in os.walk(INPUT_PATH):

    flair_path = None
    seg_path = None

    for file in files:
        if file.endswith(".nii"):
            if "flair" in file.lower():
                flair_path = os.path.join(root, file)
            elif "seg" in file.lower():
                seg_path = os.path.join(root, file)

    # ✅ Only process when both exist
    if flair_path and seg_path:
        print(f"Processing: {root}")

        try:
            flair = nib.load(flair_path).get_fdata()
            seg = nib.load(seg_path).get_fdata()

            depth = flair.shape[2]

            for i in range(depth):
                img_slice = flair[:, :, i]
                mask_slice = seg[:, :, i]

                # ❌ REMOVE FILTER (important)
                # if np.max(mask_slice) == 0:
                #     continue

                # Resize
                img_slice = cv2.resize(img_slice, (128, 128))
                mask_slice = cv2.resize(mask_slice, (128, 128))

                # Normalize image
                if np.max(img_slice) > 0:
                    img_slice = (img_slice - np.min(img_slice)) / (np.max(img_slice) - np.min(img_slice) + 1e-8)
                    img_slice = (img_slice * 255).astype(np.uint8)
                else:
                    img_slice = img_slice.astype(np.uint8)

                # Binary mask
                mask_slice = (mask_slice > 0).astype(np.uint8) * 255

                # Save
                cv2.imwrite(os.path.join(OUTPUT_IMG, f"{count}.png"), img_slice)
                cv2.imwrite(os.path.join(OUTPUT_MASK, f"{count}.png"), mask_slice)

                count += 1

                if count >= limit:
                    break

        except Exception as e:
            print(f"Error processing {root}: {e}")

    if count >= limit:
        break

print(f"\n✅ Successfully saved {count} image-mask pairs")