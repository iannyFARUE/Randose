import nibabel as nib
import numpy as np

def get_max_min_dose(nii_file):
    """
    Extracts the maximum and minimum dose values from a NIfTI (.nii.gz) dose volume.

    Parameters:
    nii_file (str): Path to the NIfTI file.

    Returns:
    tuple: (max_dose, min_dose)
    """
    # Load the NIfTI file
    img = nib.load(nii_file)
    
    # Get the dose volume data as a NumPy array
    dose_volume = img.get_fdata()

    # Compute max and min dose values
    max_dose = np.max(dose_volume)
    min_dose = np.min(dose_volume)

    return max_dose, min_dose

# Example usage
nii_file_path1 = "OpenKBP_C3D/pt_241/dose.nii.gz"  # Replace with your actual file path
nii_file_path2 = "Model_MTAS9/Prediction/pt_241/dose.nii.gz" 
max_dose1, min_dose1 = get_max_min_dose(nii_file_path1)
max_dose2, min_dose2 = get_max_min_dose(nii_file_path2)

print(f"Max Dose: {max_dose1} Gy")
print(f"Min Dose: {min_dose1} Gy")

print(f"Max Dose: {max_dose2} Gy")
print(f"Min Dose: {min_dose2} Gy")
