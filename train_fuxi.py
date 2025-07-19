# train_fuxi.py (Updated for Date Filtering, MPS, and Data Subsetting)
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import AdamW
from datetime import datetime # Import the datetime module

# The Unet3D architecture with Cuboid Attention is the core of your Custom FuXi model
from unet_cuboid import Unet3D

# --- Configuration ---
PROCESSED_DATA_DIR = './fuxi_input_features/'
EPOCHS = 100
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
INPUT_SEQUENCE_LENGTH = 12
TARGET_SEQUENCE_LENGTH = 6

# --- NEW: Date Filtering Configuration ---
# Define the date range you want to use for training.
# The format is Day-MonthAbbreviation-Year (e.g., '01APR2025')
START_DATE = "01APR2025"
END_DATE = "05APR2025"


class FuXiDataset(Dataset):
    def __init__(self, data_dir, files, input_len, target_len):
        self.data_dir = data_dir
        self.files = files
        self.total_len = input_len + target_len

    def __len__(self):
        return len(self.files) - self.total_len + 1

    def __getitem__(self, idx):
        file_sequence = self.files[idx : idx + self.total_len]
        data = [np.load(os.path.join(self.data_dir, f)) for f in file_sequence]
        
        full_sequence = np.stack(data, axis=0)
        
        mean, std = np.mean(full_sequence), np.std(full_sequence)
        full_sequence = (full_sequence - mean) / (std + 1e-6)

        input_data = torch.from_numpy(full_sequence[:INPUT_SEQUENCE_LENGTH])
        target_data = torch.from_numpy(full_sequence[INPUT_SEQUENCE_LENGTH:])
        
        assert input_data.ndim == 4, f"Input data has {input_data.ndim} dimensions, expected 4"
        
        return input_data.permute(1, 0, 2, 3), target_data.permute(1, 0, 2, 3)

def main():
    
    device = torch.device("cpu")
    print("Using CPU backend.")

    model = Unet3D(
        dim=64,
        channels=6,
        dim_mults=(1, 2, 4),
        out_frames=TARGET_SEQUENCE_LENGTH
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    all_files = sorted([f for f in os.listdir(PROCESSED_DATA_DIR) if f.endswith('.npy')])
    
    # --- CHANGE: Filter files based on the specified date range ---
    files_subset = []
    start_date_obj = datetime.strptime(START_DATE, "%d%b%Y")
    end_date_obj = datetime.strptime(END_DATE, "%d%b%Y")
    
    for filename in all_files:
        try:
            # Extract date string from filename, e.g., '01APR2025' from '3RIMG_01APR2025_...'
            date_str = filename.split('_')[1]
            file_date_obj = datetime.strptime(date_str, "%d%b%Y")
            
            # Check if the file's date is within the desired range
            if start_date_obj <= file_date_obj <= end_date_obj:
                files_subset.append(filename)
        except (IndexError, ValueError):
            # Skip files that don't match the expected naming convention
            continue

    print(f"Found {len(files_subset)} files between {START_DATE} and {END_DATE}.")
    
    if len(files_subset) < (INPUT_SEQUENCE_LENGTH + TARGET_SEQUENCE_LENGTH):
        print(f"Error: Not enough .npy files in the selected date range to create a single sequence.")
        return

    dataset = FuXiDataset(PROCESSED_DATA_DIR, files_subset, INPUT_SEQUENCE_LENGTH, TARGET_SEQUENCE_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print("Starting FuXi Model Training on the date-filtered subset...")
    for epoch in range(EPOCHS):
        model.train()
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for input_seq, target_seq in progress_bar:
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)
            
            optimizer.zero_grad()
            predicted_seq = model(input_seq)
            loss = F.mse_loss(predicted_seq, target_seq)
            loss.backward()
            optimizer.step()
            
            progress_bar.set_postfix(loss=loss.item())
        
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'fuxi_model_epoch_{epoch+1}.pth')

if __name__ == '__main__':
    main()