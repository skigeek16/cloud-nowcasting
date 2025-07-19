# inference.py
import torch
import numpy as np
import os
from tqdm import tqdm
from unet_cuboid import Unet3D

# --- Configuration ---
SATELLITE_DATA_DIR = 'FOLDER'
FUXI_FORECAST_DIR = './fuxi_forecasts_unseen/'
MODEL_PATH = './satcast_phase1_epoch_300.pth'
PAST_FRAMES = 8
FUTURE_FRAMES = 4 # How many frames to generate
FUXI_CONDITION_FRAMES = 12
TIMESTEPS = 1000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Diffusion Helpers (same as in training) ---
def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

betas = linear_beta_schedule(TIMESTEPS).to(DEVICE)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

# --- Sampling Loop ---
@torch.no_grad()
def p_sample(model, x, t, t_index, fuxi_condition):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)
    
    model_input = torch.cat([x, fuxi_condition], dim=1)
    
    pred_noise = model(model_input, t)
    
    model_mean = sqrt_recip_alphas_t * (x - betas_t * pred_noise / sqrt_one_minus_alphas_cumprod_t)

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def p_sample_loop(model, shape, fuxi_condition):
    b = shape[0]
    img = torch.randn(shape, device=DEVICE)
    imgs = []

    for i in tqdm(reversed(range(0, TIMESTEPS)), desc='Sampling time step', total=TIMESTEPS):
        img = p_sample(model, img, torch.full((b,), i, device=DEVICE, dtype=torch.long), i, fuxi_condition)
    return img

def main():
    # Load model
    model = Unet3D(dim=64, channels=1 + 6, out_dim=1, dim_mults=(1, 2, 4, 8)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # Load input data (first available sequence)
    sat_files = sorted(os.listdir(SATELLITE_DATA_DIR))
    input_sat_files = sat_files[:PAST_FRAMES]
    
    fuxi_file = sat_files[0].replace('satellite', 'fuxi_forecast')
    fuxi_path = os.path.join(FUXI_FORECAST_DIR, fuxi_file)

    # Preprocess input
    input_sat_data = [np.load(os.path.join(SATELLITE_DATA_DIR, f)) for f in input_sat_files]
    past_frames = np.stack(input_sat_data, axis=0)
    past_frames = (past_frames - np.mean(past_frames)) / (np.std(past_frames) + 1e-6)
    past_frames_tensor = torch.from_numpy(past_frames).unsqueeze(0).unsqueeze(0).to(DEVICE) # (B, C, T, H, W)
    
    fuxi_data = np.load(fuxi_path)
    fuxi_data = (fuxi_data - np.mean(fuxi_data)) / (np.std(fuxi_data) + 1e-6)
    fuxi_condition_tensor = torch.from_numpy(fuxi_data).unsqueeze(0).to(DEVICE)
    
    # We only need the part of fuxi condition that aligns with the future frames
    fuxi_condition_tensor = fuxi_condition_tensor[:, :, :FUTURE_FRAMES, :, :]
    
    print("Generating forecast...")
    output_shape = (1, 1, FUTURE_FRAMES, fuxi_condition_tensor.shape[-2], fuxi_condition_tensor.shape[-1])
    generated_frames = p_sample_loop(model, shape=output_shape, fuxi_condition=fuxi_condition_tensor)
    
    # Save result
    output_array = generated_frames.cpu().numpy()
    np.save('generated_forecast.npy', output_array)
    print("Forecast saved to generated_forecast.npy")

if __name__ == '__main__':
    main()