import os
import shutil
from transformers import BartTokenizer, BartForConditionalGeneration

# Load the model once to ensure it's cached
model_name = "facebook/bart-large-cnn"
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)

# Locate the cache directory
home = os.path.expanduser("~")
cache_base = os.path.join(home, ".cache", "huggingface", "hub", "models--facebook--bart-large-cnn", "snapshots")

# Get the first (usually only) snapshot folder
snapshot_folders = os.listdir(cache_base)
if not snapshot_folders:
    raise FileNotFoundError("No snapshot folder found. Make sure the model has been downloaded.")
snapshot_path = os.path.join(cache_base, snapshot_folders[0])  # pick the first snapshot

# Define your target directory
target_path = "./models/bart-large-cnn"

# Copy contents from cache to your local models folder
if not os.path.exists(target_path):
    shutil.copytree(snapshot_path, target_path)
    print(f"Model successfully copied to: {target_path}")
else:
    print(f"Target directory already exists: {target_path}")
