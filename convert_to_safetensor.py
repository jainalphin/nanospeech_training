import torch
from safetensors.torch import save_file
import os
import re

def get_latest_checkpoint(folder):
    pt_files = [f for f in os.listdir(folder) if f.endswith('.pt')]
    def extract_step(f):
        match = re.search(r"_(\d+)\.pt", f)
        return int(match.group(1)) if match else -1
    pt_files = sorted(pt_files, key=extract_step, reverse=True)
    return os.path.join(folder, pt_files[0]) if pt_files else None

def convert_to_safetensors(pt_path, output_path):
    print(f"Converting: {pt_path}")
    d = torch.load(pt_path, map_location="cpu")
    data = d['model_state_dict']
    for key in data:
        if isinstance(data[key], torch.Tensor):
            data[key] = data[key].cpu()

    # Ensure the parent directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    save_file(data, output_path)
    print(f"✅ Saved: {output_path}")

# Process nanospeech
nanospeech_ckpt = get_latest_checkpoint('model/nanospeech')
if nanospeech_ckpt:
    convert_to_safetensors(nanospeech_ckpt, 'models/model.safetensors')
else:
    print("No checkpoint found for nanospeech")

# Process duration
duration_ckpt = get_latest_checkpoint('model/duration')
if duration_ckpt:
    convert_to_safetensors(duration_ckpt, 'models/duration.safetensors')
else:
    print("No checkpoint found for duration")
