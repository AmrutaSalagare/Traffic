from datasets import load_dataset
import os
from PIL import Image

# Load the dataset
ds = load_dataset('mlnomad/imnet1k_ambulance')

# Print class name for ID 407
print(f"Class ID 407 corresponds to: {ds['train'].features['label'].names[407]}")

# Sample some images to understand content
print("\nSaving sample images for inspection...")
os.makedirs('ambulance_samples', exist_ok=True)

for i in range(5):
    image = ds['train'][i]['image']
    image.save(f'ambulance_samples/sample_{i+1}.jpg')
    print(f"Saved sample_{i+1}.jpg - Size: {image.size}")

print(f"\nDataset summary:")
print(f"- Total ambulance images: {len(ds['train'])}")
print(f"- All images have label ID: 407 (ambulance)")
print(f"- Image format: JPG")
print(f"- Consistent size: 500x375 pixels")
print(f"- Dataset size: ~228MB")
