import os
import torch
import sys
from safetensors.torch import save_file
from huggingface_hub import HfApi, HfFolder
from getpass import getpass
import re
import argparse


def get_hf_token():
    """
    Get Hugging Face token from different sources.

    Returns
    -------
    str
        The Hugging Face token.

    Notes
    -----
    Tries to get token from:
    1. Kaggle secrets
    2. Environment variable
    3. User input
    """
    # Try Kaggle secrets
    try:
        from kaggle_secrets import UserSecretsClient
        hf_token = UserSecretsClient().get_secret("HF_TOKEN")
        if hf_token:
            print("✅ Using Hugging Face token from Kaggle secrets")
            return hf_token
    except ImportError:
        pass

    # Try environment variable
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        print("✅ Using Hugging Face token from environment variable")
        return hf_token

    # Ask user for token
    hf_token = getpass("Enter your Hugging Face token: ")
    if not hf_token:
        print("❌ No Hugging Face token provided. Exiting.")
        sys.exit(1)

    return hf_token


def get_latest_checkpoint(folder):
    """
    Find the latest checkpoint file in the specified folder.

    Parameters
    ----------
    folder : str
        Directory path containing checkpoint files.

    Returns
    -------
    str or None
        Path to the latest checkpoint file or None if no checkpoints found.
    """
    if not os.path.exists(folder):
        print(f"⚠️ Checkpoint folder not found: {folder}")
        return None

    pt_files = [f for f in os.listdir(folder) if f.endswith('.pt')]
    if not pt_files:
        print(f"⚠️ No checkpoint files found in: {folder}")
        return None

    def extract_step(f):
        match = re.search(r"_(\d+)\.pt", f)
        return int(match.group(1)) if match else -1

    pt_files = sorted(pt_files, key=extract_step, reverse=True)
    latest_file = os.path.join(folder, pt_files[0])
    print(f"📄 Found latest checkpoint: {latest_file}")
    return latest_file


def convert_to_safetensors(pt_path, output_path):
    """
    Convert PyTorch checkpoint to safetensors format.

    Parameters
    ----------
    pt_path : str
        Path to the PyTorch checkpoint file.
    output_path : str
        Path where safetensors file will be saved.

    Returns
    -------
    bool
        True if conversion was successful, False otherwise.
    """
    try:
        print(f"🔄 Converting {pt_path} to safetensors...")
        d = torch.load(pt_path, map_location="cpu")

        if 'model_state_dict' not in d:
            print(f"❌ Invalid checkpoint format: 'model_state_dict' not found in {pt_path}")
            return False

        data = d['model_state_dict']
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].cpu()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        save_file(data, output_path)
        print(f"✅ Saved: {output_path}")
        return True
    except Exception as e:
        print(f"❌ Error converting checkpoint: {str(e)}")
        return False


def upload_to_huggingface(model_space_name, files, hf_token):
    """
    Upload files to Hugging Face model repository.

    Parameters
    ----------
    model_space_name : str
        Name of the model repository.
    files : list
        List of file paths to upload.
    hf_token : str
        Hugging Face authentication token.

    Returns
    -------
    bool
        True if upload was successful, False otherwise.
    """
    if not files:
        print("❌ No files to upload.")
        return False

    try:
        api = HfApi()
        user_info = api.whoami(token=hf_token)
        repo_id = f"{user_info['name']}/{model_space_name}"

        # Create repo if not exists
        try:
            api.repo_info(repo_id, token=hf_token)
            print(f"ℹ️ Repository exists: {repo_id}")
        except Exception as e:
            print(f"📁 Creating repository: {repo_id}")
            api.create_repo(repo_id, token=hf_token, private=False)

        # Upload each file
        for f in files:
            print(f"⬆️ Uploading: {f}")
            api.upload_file(
                path_or_fileobj=f,
                path_in_repo=os.path.basename(f),
                repo_id=repo_id,
                token=hf_token
            )

        print(f"✅ All files uploaded to: https://huggingface.co/{repo_id}")
        return True
    except Exception as e:
        print(f"❌ Error uploading to Hugging Face: {str(e)}")
        return False


def parse_arguments():
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Convert and upload model checkpoints to Hugging Face")

    parser.add_argument("--nanospeech_dir", type=str, default="model/nanospeech",
                        help="Directory containing nanospeech model checkpoints")
    parser.add_argument("--duration_dir", type=str, default="model/duration",
                        help="Directory containing duration model checkpoints")
    parser.add_argument("--output_dir", type=str, default="models",
                        help="Output directory for safetensors files")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Name for the Hugging Face model repository")

    return parser.parse_args()


def main():
    """
    Main function to handle model conversion and upload.
    """
    args = parse_arguments()

    # Get Hugging Face token
    hf_token = get_hf_token()
    if not hf_token:
        return

    # Set up HF folder
    HfFolder.save_token(hf_token)

    # Ask for model name if not provided as argument
    model_space_name = args.model_name
    if not model_space_name:
        print("❌ Model space name is required. Exiting.")
        return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Convert checkpoints
    output_files = []

    # Process nanospeech model
    nanospeech_ckpt = get_latest_checkpoint(args.nanospeech_dir)
    if nanospeech_ckpt:
        out_path = os.path.join(args.output_dir, "model.safetensors")
        if convert_to_safetensors(nanospeech_ckpt, out_path):
            output_files.append(out_path)

    # Process duration model
    duration_ckpt = get_latest_checkpoint(args.duration_dir)
    if duration_ckpt:
        out_path = os.path.join(args.output_dir, "duration.safetensors")
        if convert_to_safetensors(duration_ckpt, out_path):
            output_files.append(out_path)

    # Upload to Hugging Face
    output_files.append("vocab.txt")
    if output_files:
        upload_to_huggingface(model_space_name, output_files, hf_token)
    else:
        print("❌ No files were successfully converted. Nothing to upload.")


if __name__ == "__main__":
    main()
