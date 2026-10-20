import os
import yaml

# 1. Read the paths dynamically from config.yaml
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

local_path = config['training']['training_data_dir']
kaggle_path = config['training']['kaggle_data_dir']

# 2. Detect the environment
is_kaggle = "KAGGLE_KERNEL_RUN_TYPE" in os.environ
target_path = kaggle_path if is_kaggle else local_path
inactive_path = local_path if is_kaggle else kaggle_path

# 3. Update dvc.yaml to track the correct environment path
with open("dvc.yaml", "r") as f:
    dvc_content = f.read()

# Swap the old path for the active one
if inactive_path in dvc_content:
    dvc_content = dvc_content.replace(inactive_path, target_path)
    with open("dvc.yaml", "w") as f:
        f.write(dvc_content)

print(f"✅ Environment configured for: {'Kaggle' if is_kaggle else 'Local'}")
print(f"📂 dvc.yaml dataset dependency set to: {target_path}")