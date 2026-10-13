import os
import pandas as pd

def verify_isic_images(csv_path: str, directories_to_check: list):
    """
    Verifies if images in a list of directories exist in the provided ISIC Ground Truth CSV.
    """
    # 1. Load the Ground Truth CSV
    print(f"Loading Ground Truth data from: {os.path.basename(csv_path)}...")
    try:
        df = pd.read_csv(csv_path)
        # Convert the 'image' column to a Python Set for instant lookups
        valid_names_set = set(df['image'].astype(str))
        print(f"✅ Loaded {len(valid_names_set)} valid image names.\n")
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return

    # 2. Process each directory provided in the list
    for folder_path in directories_to_check:
        folder_name = os.path.basename(os.path.normpath(folder_path))
        
        if not os.path.exists(folder_path):
            print(f"❌ Directory not found: {folder_path}\n" + "-" * 40)
            continue

        # Grab all image files in the directory
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        total_files = len(files)
        
        if total_files == 0:
            print(f"⚠️ No images found in '{folder_name}' folder.\n" + "-" * 40)
            continue

        match_count = 0
        unmatched_files = []

        # 3. Check each file against the CSV data
        for file in files:
            name_without_ext = os.path.splitext(file)[0]
            
            if name_without_ext in valid_names_set:
                match_count += 1
            else:
                unmatched_files.append(file)

        # 4. Print Statistics for the current folder
        match_percentage = (match_count / total_files) * 100
        print(f"--- Results for '{folder_name}' ---")
        print(f"Total images found: {total_files}")
        print(f"Matches CSV data:   {match_count} ({match_percentage:.2f}%)")
        print(f"Unknown images:     {len(unmatched_files)}")
        
        if unmatched_files:
            print(f"Examples of unknown images: {unmatched_files[:5]}")
        print("-" * 40 + "\n")

# ==========================================
# HOW TO USE THE FUNCTION
# ==========================================
if __name__ == "__main__":
    # Define your paths
    truth_data_file = r"K:\projects\skin-disease-classification-system-model-with-79%-accuracy\artifacts\data_source\ISIC_2019_Training_Inputs\ISIC_2019_Training_Input\ISIC_2019_Training_GroundTruth.csv"
    
    # You can add as many folders as you want to this list
    folders_to_verify = [
        r"K:\projects\skin-disease-classification-system-model-with-79%-accuracy\artifacts\data_source\skin-disease-images\Affected",
        r"K:\projects\skin-disease-classification-system-model-with-79%-accuracy\artifacts\data_source\skin-disease-images\Healthy"
    ]
    
    # Run the verification
    verify_isic_images(csv_path=truth_data_file, directories_to_check=folders_to_verify)