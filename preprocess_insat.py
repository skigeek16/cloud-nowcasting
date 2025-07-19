# preprocess_insat.py (Final Version - Reads full image and applies conversion)
import h5py
import numpy as np
import os
from scipy.ndimage import uniform_filter

# --- Conversion Constants ---
# IMPORTANT: These are placeholder values. You MUST find the correct
# slope and offset for each channel from your INSAT-3D data documentation.
CONVERSION_CONSTANTS = {
    'TIR1': {'slope': 0.005, 'offset': -2.0},
    'TIR2': {'slope': 0.005, 'offset': -2.0},
    'WV':   {'slope': 0.005, 'offset': -2.0},
    'MIR':  {'slope': 0.01,  'offset': -5.0},
    'VIS':  {'slope': 0.1,   'offset': 0.0},
}

# --- Helper functions ---
def convert_to_physical_values(counts, channel_name):
    """Converts raw counts to physical values (radiance/albedo) using slope and offset."""
    consts = CONVERSION_CONSTANTS[channel_name]
    slope, offset = consts['slope'], consts['offset']
    # Formula is often Radiance = (Count * Slope) + Offset
    return (counts * slope) + offset

def get_brightness_temp_from_radiance(radiance, channel_name):
    """Converts radiance to brightness temperature using Planck's inverse function."""
    planck_consts = {
        'MIR':  {'c1': 1.191042e8, 'c2': 1.43877e4, 'nu': 2564.1},
        'TIR1': {'c1': 1.191042e8, 'c2': 1.43877e4, 'nu': 925.9}, # For 10.8 µm
        'TIR2': {'c1': 1.191042e8, 'c2': 1.43877e4, 'nu': 833.3}, # For 12.0 µm
        'WV':   {'c1': 1.191042e8, 'c2': 1.43877e4, 'nu': 1470.6} # For 6.8 µm
    }
    if channel_name not in planck_consts:
        raise ValueError(f"Planck constants not defined for channel: {channel_name}")
    
    consts = planck_consts[channel_name]
    c1, c2, nu = consts['c1'], consts['c2'], consts['nu']
    
    safe_radiance = np.maximum(radiance, 1e-9)
    temp = (c2 * nu) / np.log(1 + (c1 * nu**3) / safe_radiance)
    return temp

# --- Cloud Detection and CTT Retrieval (Same as before) ---
def primary_dynamic_threshold_test(bt_tir1, land_mask):
    bts = uniform_filter(bt_tir1, size=15) + 5
    ocean_threshold, land_threshold = bts - (0.03 * bts), bts - (0.05 * bts)
    is_cloudy = np.zeros_like(bt_tir1, dtype=bool)
    is_cloudy[~land_mask] = bt_tir1[~land_mask] < ocean_threshold[~land_mask]
    is_cloudy[land_mask] = bt_tir1[land_mask] < land_threshold[land_mask]
    return is_cloudy

def retrieve_ctt(cloud_mask, bt_tir1):
    ctt = np.full_like(bt_tir1, 300.0) # Default to a warm surface temp
    ctt[cloud_mask] = bt_tir1[cloud_mask]
    ctt[~cloud_mask] = 0 # Use 0 to signify no cloud
    return ctt

def create_final_features(ctt, bt_wv, cloud_mask):
    clear_sky_temp = 295.0
    ctt[~cloud_mask] = clear_sky_temp
    bt_wv[~cloud_mask] = 240.0
    proxy_features = {
        'SST': ctt, 'CTT': ctt, 'OLR': ctt,
        'UTH': bt_wv, 'TPW': bt_wv, 'CMV': np.zeros_like(ctt)
    }
    return np.stack(list(proxy_features.values()), axis=0).astype(np.float32)

def process_l1c_file(filepath, land_mask):
    print(f"Processing {filepath}...")
    with h5py.File(filepath, 'r') as f:
        # 1. Load the FULL IMAGE data counts
        counts_tir1 = f['IMG_TIR1'][0, :, :]
        counts_tir2 = f['IMG_TIR2'][0, :, :]
        counts_wv = f['IMG_WV'][0, :, :]
        counts_mir = f['IMG_MIR'][0, :, :]

        # 2. Convert counts to physical values (Radiance)
        rad_tir1 = convert_to_physical_values(counts_tir1, 'TIR1')
        rad_tir2 = convert_to_physical_values(counts_tir2, 'TIR2')
        rad_wv = convert_to_physical_values(counts_wv, 'WV')
        rad_mir = convert_to_physical_values(counts_mir, 'MIR')
        
        # 3. Convert radiance to Brightness Temperature
        bt_tir1 = get_brightness_temp_from_radiance(rad_tir1, 'TIR1')
        bt_tir2 = get_brightness_temp_from_radiance(rad_tir2, 'TIR2')
        bt_wv = get_brightness_temp_from_radiance(rad_wv, 'WV')
        
        # 4. Generate Cloud Mask using the physics-based tests
        final_cloud_mask = primary_dynamic_threshold_test(bt_tir1, land_mask)
        # Add other tests from the paper here if desired for more accuracy

        # 5. Retrieve Cloud Top Temperature (CTT)
        ctt = retrieve_ctt(final_cloud_mask, bt_tir1)

        # 6. Create Final 6-Channel Feature Stack
        final_features = create_final_features(ctt, bt_wv, final_cloud_mask)
        final_features = np.nan_to_num(final_features)
        
        return final_features

def main(raw_dir, processed_dir):
    os.makedirs(processed_dir, exist_ok=True)
    files = sorted([f for f in os.listdir(raw_dir) if f.endswith('.h5') or f.endswith('.H5')])
    
    if not files:
        print(f"Error: No .h5 files found in '{raw_dir}'.")
        return

    # Create a dummy land mask for now
    with h5py.File(os.path.join(raw_dir, files[0]), 'r') as f:
        h, w = f['IMG_TIR1'].shape[1], f['IMG_TIR1'].shape[2]
        land_sea_mask = np.zeros((h, w), dtype=bool)

    for filename in files:
        raw_path = os.path.join(raw_dir, filename)
        try:
            processed_features = process_l1c_file(raw_path, land_sea_mask)
            output_filename = filename.rsplit('.', 1)[0] + '.npy'
            output_path = os.path.join(processed_dir, output_filename)
            np.save(output_path, processed_features)
            print(f"Saved processed features to {output_path}")
        except Exception as e:
            print(f"An unexpected error occurred with file {filename}: {e}")
            # Optional: Uncomment the next line to stop on error
            # raise e

if __name__ == '__main__':
    RAW_L1C_DATA_DIR = 'FOLDER'
    PROCESSED_FUXI_INPUT_DIR = './fuxi_input_features/'
    
    print("Starting INSAT-3D L1C data preprocessing with full image data...")
    main(RAW_L1C_DATA_DIR, PROCESSED_FUXI_INPUT_DIR)
    print("Preprocessing complete.")