#!/usr/bin/env python3
import os
import pickle
import random
import concurrent.futures
import torch
import rasterio as rio
from rasterio.transform import xy

# --- Define only the functions needed for preprocessing ---

def bands_coord_image(file_path):
    """
    Reads a GeoTIFF file and extracts bands plus the image center coordinates.
    Assumes the file has at least 6 bands.
    """
    with rio.open(file_path) as img:
        # Read and transpose from (bands, H, W) to (H, W, bands)
        img_array = img.read().transpose(1, 2, 0)
        rgb = img_array[:, :, 0:3]
        infrared = img_array[:, :, 3]
        elevation = img_array[:, :, 4]
        target_labels = img_array[:, :, 5]
        width, height = img.width, img.height
        center_x = width // 2
        center_y = height // 2
        lon, lat = xy(img.transform, center_y, center_x)
    return rgb, infrared, elevation, target_labels, lon, lat

def fancy_transform(sample):
    """
    Converts raw arrays into normalized tensors.
    Returns a dictionary with keys: 'rgb', 'infrared', 'elevation', 'labels', 'longitude', 'latitude'.
    """
    sample["rgb"] = torch.as_tensor(sample["rgb"]).permute(2, 0, 1).float().div_(255)
    sample["infrared"] = torch.as_tensor(sample["infrared"]).unsqueeze(0).float().div_(255)
    sample["elevation"] = torch.as_tensor(sample["elevation"]).unsqueeze(0).float().div_(255)
    return sample

def process_file(file_path):
    """
    Processes a single GeoTIFF file: reads the data, builds a dictionary,
    and applies the transformation.
    """
    rgb, infrared, elevation, target_labels, lon, lat = bands_coord_image(file_path)
    data = {
        "rgb": rgb,
        "infrared": infrared,
        "elevation": elevation,
        "labels": target_labels,
        "longitude": lon,
        "latitude": lat
    }
    return fancy_transform(data)

# --- Main function ---
def main():
    # Set the directory containing your GeoTIFF files.
    data_dir = "Potsdam-GeoTif/Potsdam-GeoTif"  # change this to your actual directory path
    # List all .tif files in the directory.
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.tif')]
    print(f"Found {len(files)} GeoTIFF files.")
    
    # Randomly select a subset (e.g., 5000 files) for processing.
    number_of_subset = 15048
    subset_files = random.sample(files, number_of_subset)
    print(f"Processing a subset of {len(subset_files)} files.")

    # Use all available cores.
    num_workers = os.cpu_count()
    print(f"Detected {num_workers} CPU cores.")
    print(f"Using {num_workers} workers for parallel processing.")

    # Process files in parallel.
    preprocessed_data = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(process_file, subset_files))
    
    for idx, sample in enumerate(results):
        file_path = subset_files[idx]
        preprocessed_data[file_path] = sample
        if idx % 100 == 0:
            print(f"Processed {idx} files out of {len(subset_files)}")
    
    # Save the preprocessed data to a pickle file.
    output_path = "preprocessed_data_15048.pkl"  # or specify an absolute path if desired
    with open(output_path, "wb") as f:
        pickle.dump(preprocessed_data, f)
    
    print("Preprocessing complete and data saved to:", output_path)

if __name__ == "__main__":
    main()
