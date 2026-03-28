import numpy as np
import cv2
import os

def generate_brats_like_data(count=15, size=(128, 128)):
    """
    Generates synthetic data that mimics BraTS 2020 FLAIR slices for demonstration.
    """
    img_dir = 'backend/dataset/images'
    mask_dir = 'backend/dataset/masks'
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    print(f"Generating {count} BraTS-like synthetic MRI slices...")

    for i in range(count):
        # Create a brain-like shape (anatomical base)
        img = np.zeros(size, dtype=np.uint8)
        # Skull/Brain outer boundary
        cv2.ellipse(img, (64, 64), (50, 45), 0, 0, 360, 30, -1)
        # Inner brain structure (simulating ventricles/folds)
        cv2.ellipse(img, (64, 64), (45, 40), 0, 0, 360, 50, -1)
        
        # Add some "FLAIR-like" intensity variation
        noise = np.random.randint(0, 40, size, dtype=np.uint8)
        img = cv2.add(img, noise)
        
        # Create a tumor-like blob for the mask
        mask = np.zeros(size, dtype=np.uint8)
        # Random location within the brain area
        tx, ty = np.random.randint(40, 88, 2)
        tr = np.random.randint(8, 20)
        
        # Create an irregular tumor shape
        num_points = 8
        angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
        radii = tr + np.random.randint(-5, 5, num_points)
        pts = np.array([ [tx + r*np.cos(a), ty + r*np.sin(a)] for r, a in zip(radii, angles) ], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
        
        # Smooth the mask a bit
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # Add the tumor to the image as a bright, hyper-intense region (typical of FLAIR)
        # Also add some edema (subtle halo around tumor)
        edema = cv2.GaussianBlur(mask, (15, 15), 0)
        img = cv2.addWeighted(img, 1.0, edema, 0.3, 0)
        img = cv2.addWeighted(img, 1.0, mask, 0.6, 0)
        
        # Add final noise
        final_noise = np.random.normal(0, 5, size).astype(np.uint8)
        img = cv2.add(img, final_noise)

        # Normalize to 0-1 range internally but save as uint8 for storage
        # The user wants "Normalize (0-1)" but cv2.imwrite usually takes uint8
        # Training code will handle the 0-1 normalization.
        
        cv2.imwrite(os.path.join(img_dir, f'img_{i}.png'), img)
        cv2.imwrite(os.path.join(mask_dir, f'mask_{i}.png'), mask)

if __name__ == '__main__':
    generate_brats_like_data()
    print("Dataset ready in backend/dataset/")
