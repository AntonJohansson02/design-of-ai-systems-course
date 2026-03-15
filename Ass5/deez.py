import os
import pickle
import random
import concurrent.futures
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import rasterio as rio
from rasterio.transform import xy


# ----- Preprocessing Functions -----

def bands_coord_image(file_path):
    """
    Reads a GeoTIFF file and extracts bands + coordinates
    """
    with rio.open(file_path) as img:
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

def simple_transform(sample):
    sample["rgb"] = torch.as_tensor(sample["rgb"]).permute(2, 0, 1).float().div(255)
    sample["infrared"] = torch.as_tensor(sample["infrared"]).unsqueeze(0).float().div(255)
    sample["elevation"] = torch.as_tensor(sample["elevation"]).unsqueeze(0).float().div(255)
    return sample

def process_file(file_path):
    """
    Processes a single GeoTIFF file --> makes dictionary and applies transformation
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
    return simple_transform(data)

def preprocess_data(data_dir, number_of_subset, output_path, num_workers):
    """
    Processes all the GeoTIFF files in parallel --> makes single pickle file
    """
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.tif')]
    print(f"Found {len(files)} GeoTIFF files.")
    subset_files = random.sample(files, number_of_subset)
    print(f"Processing a subset of {len(subset_files)} files.")
    print(f"Using {num_workers} workers for parallel processing.")

    preprocessed_data = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(process_file, subset_files))
    
    for idx, sample in enumerate(results):
        file_path = subset_files[idx]
        preprocessed_data[file_path] = sample
        if idx % 100 == 0:
            print(f"Processed {idx} files out of {len(subset_files)}")
    
    with open(output_path, "wb") as f:
        pickle.dump(preprocessed_data, f)
    
    print("Preprocessing complete and data saved to:", output_path)
    return preprocessed_data

# ----- Dataset Definition -----

class PreprocessedDataset(Dataset):
    def __init__(self, data_dict):
        self.data_dict = data_dict
        # Create a list of keys for indexing.
        self.keys = list(data_dict.keys())
        
    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):
        key = self.keys[idx]
        return self.data_dict[key]

# ----- Model Definition -----

class Segmentation(nn.Module):
    def __init__(self, num_classes):
        super(Segmentation, self).__init__()
        # Input: RGB image, so 3 channels.
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, num_classes, kernel_size=3, padding=1)
        
    def forward(self, x):
        # x: [batch, 3, H, W]
        x = F.relu(self.conv1(x))  # -> [batch, 16, H, W]
        x = self.conv2(x)          # -> [batch, num_classes, H, W]
        return x

# ----- Training Function -----

def train_model(preprocessed_data, num_epochs=10, batch_size=16, num_workers=8):
    # Create dataset from preprocessed data
    dataset = PreprocessedDataset(preprocessed_data)
    num_samples = len(dataset)
    fold_size = num_samples // 5
    indices = list(range(num_samples))
    folds_indices = [indices[i*fold_size:(i+1)*fold_size] for i in range(5)]

    # Use folds 1,2,3 for training, fold 4 for validation, and fold 5 for testing.
    train_indices = folds_indices[0] + folds_indices[1] + folds_indices[2]
    val_indices   = folds_indices[3]
    test_indices  = folds_indices[4]

    train_dataset = Subset(dataset, train_indices)
    val_dataset   = Subset(dataset, val_indices)
    test_dataset  = Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Segmentation(num_classes=6)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            inputs = batch['rgb'].to(device)
            labels = batch['labels'].to(device).long()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            
        
        # Evaluate on validation set
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['rgb'].to(device)
                labels = batch['labels'].to(device).long()
                outputs = model(inputs)
                val_loss += loss_fn(outputs, labels).item()
        print(f"Epoch {epoch}, Validation Loss: {val_loss}")

    print("Training complete.")

# ----- Main Workflow -----

def main():
    # Parameters for preprocessing and training
    data_dir = "Potsdam-GeoTif/Potsdam-GeoTif"  # Update to your directory path
    number_of_subset = 15048
    pickle_path = 'C:/Users/anton/OneDrive - Chalmers/Läsår 4/Design of AI systems - DAT410/Assignments/Ass5/preprocessed_data_5000.pkl'
    num_workers = os.cpu_count()

    # Preprocess data if pickle does not exist, else load it.
    if not os.path.exists(pickle_path):
        preprocessed_data = preprocess_data(data_dir, number_of_subset, pickle_path, num_workers)
    else:
        with open(pickle_path, 'rb') as f:
            preprocessed_data = pickle.load(f)
        print("Loaded preprocessed data from pickle.")

    # Train the segmentation model using the preprocessed data.
    train_model(preprocessed_data)

if __name__ == "__main__":
    main()