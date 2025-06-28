import numpy as np
import os
import SimpleITK as sitk
from tqdm import tqdm

# Function to compute mean absolute dose difference
def get_3D_Dose_dif(pred, gt, possible_dose_mask=None):
    if possible_dose_mask is not None:
        pred = pred[possible_dose_mask > 0]
        gt = gt[possible_dose_mask > 0]

    return np.mean(np.abs(pred - gt))

# Function to process all patients and compute dose differences
def get_Dose_score_and_DVH_score(prediction_dir, gt_dir):
    patient_scores = {}

    # Get patient folders from the prediction directory
    list_patient_ids = [f for f in os.listdir(prediction_dir) if os.path.isdir(os.path.join(prediction_dir, f))]

    for patient_id in tqdm(list_patient_ids, desc="Processing Patients"):
        pred_path = os.path.join(prediction_dir, patient_id, "dose.nii.gz")
        gt_path = os.path.join(gt_dir, patient_id, "dose.nii.gz")
        mask_path = os.path.join(gt_dir, patient_id, "possible_dose_mask.nii.gz")

        if not (os.path.exists(pred_path) and os.path.exists(gt_path) and os.path.exists(mask_path)):
            print(f"Skipping {patient_id}: Missing files.")
            continue

        # Load data
        pred_nii = sitk.ReadImage(pred_path)
        pred = sitk.GetArrayFromImage(pred_nii)

        gt_nii = sitk.ReadImage(gt_path)
        gt = sitk.GetArrayFromImage(gt_nii)

        possible_dose_mask_nii = sitk.ReadImage(mask_path)
        possible_dose_mask = sitk.GetArrayFromImage(possible_dose_mask_nii)

        # Compute dose difference
        dose_dif = get_3D_Dose_dif(pred, gt, possible_dose_mask)

        # Store patient performance
        patient_scores[patient_id] = dose_dif

    return patient_scores

# Function to find the best and worst samples based on performance
def get_best_worst_samples(prediction_dir, gt_dir):
    patient_scores = get_Dose_score_and_DVH_score(prediction_dir, gt_dir)

    # Sort patients based on dose difference (lower is better)
    sorted_patients = sorted(patient_scores.items(), key=lambda x: x[1])

    # Get the best and worst 5
    best_5 = sorted_patients[:5]  # Top 5 best-performing
    worst_5 = sorted_patients[-5:]  # Bottom 5 worst-performing

    return best_5, worst_5

# Example usage
if __name__ == "__main__":
    prediction_dir = "Model/Prediction"
    gt_dir = "OpenKBP_C3D"

    best_5, worst_5 = get_best_worst_samples(prediction_dir, gt_dir)

    print("\nTop 5 Best Performing Patients:")
    for patient, score in best_5:
        print(f"{patient}: {score:.4f}")

    print("\nTop 5 Worst Performing Patients:")
    for patient, score in worst_5:
        print(f"{patient}: {score:.4f}")
