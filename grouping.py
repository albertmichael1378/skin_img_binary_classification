import pandas as pd
import os
import shutil

# Define your paths
csv_path = r"K:\projects\skin-disease-classification-system-model-with-79%-accuracy\artifacts\data_source\ISIC_2019_Training_Inputs\ISIC_2019_Training_Input\ISIC_2019_Training_GroundTruth.csv"
image_source_dir = r"K:\projects\skin-disease-classification-system-model-with-79%-accuracy\artifacts\data_source\ISIC_2019_Training_Inputs\ISIC_2019_Training_Input"

# Define output directories
output_dir = r"K:\projects\skin-disease-classification-system-model-with-79%-accuracy\artifacts\data_source\Binary_Classification"
cancerous_dir = os.path.join(output_dir, "cancerous")
non_cancerous_dir = os.path.join(output_dir, "non_cancerous")

# Create output folders if they don't exist
os.makedirs(cancerous_dir, exist_ok=True)
os.makedirs(non_cancerous_dir, exist_ok=True)

# Define the classes
cancerous_classes = ['MEL', 'BCC', 'SCC']
non_cancerous_classes = ['NV', 'BKL', 'DF', 'VASC', 'AK', 'UNK']

# Read the ground truth CSV
print("Loading ground truth CSV...")
df = pd.read_csv(csv_path)

missing_files = 0
copied_files = 0

print("Starting file sorting...")
for index, row in df.iterrows():
    image_name = row['image']
    image_file = image_name + ".jpg"
    source_path = os.path.join(image_source_dir, image_file)
    
    if not os.path.exists(source_path):
        missing_files += 1
        continue
        
    # Determine which class is 1.0
    # Drop the 'image' column and find the index of the max value (which will be 1.0)
    lesion_class = row.drop('image').astype(float).idxmax()
    
    # Route to the appropriate folder
    if lesion_class in cancerous_classes:
        destination_path = os.path.join(cancerous_dir, image_file)
    else:
        destination_path = os.path.join(non_cancerous_dir, image_file)
        
    # Copy the file (using copy2 to preserve metadata). Use shutil.move() if you want to move instead of copy.
    shutil.copy2(source_path, destination_path)
    copied_files += 1
    
    # Print progress every 1000 files
    if copied_files % 1000 == 0:
        print(f"Copied {copied_files} files...")

print(f"Sorting complete! Successfully copied {copied_files} files. Missing files: {missing_files}.")