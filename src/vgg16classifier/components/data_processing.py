import numpy as np
import tensorflow as tf
from keras.applications import ResNet50
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

class Data_preprocessing:
    def __init__(self):
        # 1. Hardware Configuration & GPU Detection
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ GPU detected: {len(gpus)} available. Enabling high-speed batching.")
            self.batch_size = 64
        else:
            print("⚠️ No GPU detected. Falling back to CPU batch processing.")
            self.batch_size = 32


        # 2. Setup Directories & Paths
        image_folder1 = Path("artifacts/data_source/Binary_Classification/cancerous")
        image_folder2 = Path("artifacts/data_source/Binary_Classification/non_cancerous")
        save_folder = Path("artifacts/image_test")
        # Create the folder if it does not exist (parents=True creates intermediate folders if needed)
        save_folder.mkdir(parents=True, exist_ok=True)
        self.save_path = save_folder / "resnet_embeddings.npz"
    
        # Grab all image paths (update "*.jpg" if you use pngs)
        image_paths1 = list(image_folder1.glob("*.jpg"))
        image_paths2 = list(image_folder2.glob("*.jpg"))
        self.path_strings = [str(p) for p in image_paths1 + image_paths2]
        if self.path_strings:
            print(f"Found {len(image_paths1)} images in jpg format.")
            print(f"Found {len(image_paths2)} images in jpg format.")
            print(f"Found {len(self.path_strings)} images from {image_folder1} and {image_folder2}.")
        else:
            print(f"Found {len(self.path_strings)} images from {image_folder1} and {image_folder2}.")
            print("⚠️ No images found. Please check the directory paths and image formats.")
            raise FileNotFoundError
            

    # 3. Build the Data Pipeline (The Professional Batching Method)
    def load_and_preprocess(self,path):
        """Loads and formats image tensors efficiently."""
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3) 
        img = tf.image.resize(img, [224, 224])
        # Apply ResNet50's specific mathematical scaling
        img = tf.keras.applications.resnet50.preprocess_input(img)
        return img
    def extract_embedding_matrix(self):
        
        # Create an optimized TensorFlow dataset
        dataset = tf.data.Dataset.from_tensor_slices(self.path_strings)
        # AUTOTUNE automatically determines how many CPU threads to use for loading
        dataset = dataset.map(lambda path: self.load_and_preprocess(path), num_parallel_calls=tf.data.AUTOTUNE)
        # Group into batches and prepare the next batch while the current one processes
        dataset = dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        # 4. Extract Embeddings
        print(f"Extracting embeddings using batch size {self.batch_size}...")
        model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

        # model.predict automatically consumes the batched dataset
        embedding_matrix = model.predict(dataset)

        # 5. Save the Matrix & Paths
        print(f"Saving embeddings and paths to {self.save_path}...")
        # We save the paths alongside the matrix so row 5 in the matrix always maps to row 5 in the paths
        np.savez(self.save_path, matrix=embedding_matrix, paths=self.path_strings)

        # 6. Calculate Similarity & Find Duplicates
        print("Calculating similarities...")
        similarity_matrix = cosine_similarity(embedding_matrix)

        threshold = 0.98
        duplicates_found = set()

        # Loop through the upper triangle of the matrix to avoid comparing A to B, then B to A
        for i in range(len(self.path_strings)):
            for j in range(i + 1, len(self.path_strings)):
                if similarity_matrix[i, j] > threshold:
                    print(f"Match ({similarity_matrix[i, j]:.3f}): {self.path_strings[i]} <-> {self.path_strings[j]}")
                    duplicates_found.add(self.path_strings[j])

        print(f"\nFound {len(duplicates_found)} duplicate images to remove.")