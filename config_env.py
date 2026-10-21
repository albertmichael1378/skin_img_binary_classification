import os
import sys
from pathlib import Path

# Add src to the Python path so it can import your package
sys.path.append(str(Path(__file__).parent / "src"))

from vgg16classifier.config.configuration import ConfigurationManager

# 1. Read the paths dynamically using your existing manager
config_manager = ConfigurationManager()
training_config = config_manager.config.training

local_path = training_config.training_data_dir
kaggle_path = training_config.kaggle_data_dir

# 2. Detect the environment
is_kaggle = "KAGGLE_KERNEL_RUN_TYPE" in os.environ
target_path = kaggle_path if is_kaggle else local_path
inactive_path = local_path if is_kaggle else kaggle_path

# 3. Update dvc.yaml to track the correct environment path
with open("dvc.yaml", "r") as f:
    dvc_content = f.read()

if inactive_path in dvc_content:
    dvc_content = dvc_content.replace(inactive_path, target_path)
    with open("dvc.yaml", "w") as f:
        f.write(dvc_content)

print(f"✅ Environment configured for: {'Kaggle' if is_kaggle else 'Local'}")
print(f"📂 dvc.yaml dataset dependency set to: {target_path}")