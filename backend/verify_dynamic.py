import numpy as np
import cv2
import torch
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), 'backend'))

from app.routes.inference import _simulate_multimodal, _segment_mask, _survival_predict, _postprocess_mask

def test_dynamic_outputs():
    print("Testing dynamic outputs...")
    
    # Create a dummy grayscale brain image
    img = np.zeros((128, 128), dtype=np.float32)
    # Background
    img[20:100, 20:100] = 0.2
    # Tumor-like region
    img[40:70, 40:70] = 0.8
    
    # Run multiple times with SAME input
    results_static_input = []
    print("\nRunning multiple times with SAME input to test stochastic simulation...")
    for i in range(3):
        mask = _postprocess_mask(_segment_mask(img))
        prob, days = _survival_predict(img, mask)
        results_static_input.append(days)
        print(f"Iteration {i+1} (Same Input): {days} days, Prob: {prob:.4f}")

    # Run with SLIGHTLY different input
    print("\nRunning with slightly DIFFERENT input to test dynamic range...")
    img_alt = img.copy()
    img_alt[50:80, 50:80] = 0.9 # Slightly larger/brighter tumor
    results_alt_input = []
    for i in range(3):
        mask_alt = _postprocess_mask(_segment_mask(img_alt))
        prob_alt, days_alt = _survival_predict(img_alt, mask_alt)
        results_alt_input.append(days_alt)
        print(f"Iteration {i+1} (Alt Input): {days_alt} days, Prob: {prob_alt:.4f}")

    # Assertion logic (printed)
    if results_static_input[0] != results_static_input[1] or \
       results_static_input[1] != results_static_input[2]:
        print("\n✅ PASSED: Stochasticity confirmed for same input.")
    else:
        print("\n⚠️ WARNING: Outputs are identical for same input. Check stochasticity logic.")

    if np.mean(results_static_input) != np.mean(results_alt_input):
        print("✅ PASSED: Different inputs produced different mean outputs.")
    else:
        print("⚠️ WARNING: Different inputs produced identical mean outputs.")

if __name__ == "__main__":
    test_dynamic_outputs()
