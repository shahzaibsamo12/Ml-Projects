from transformers import pipeline
pipeline("summarization", model="facebook/bart-large-cnn", cache_dir="./models")
print("cache_dir")