from datasets import load_dataset
import os

# Load the ambulance dataset
print("Loading ambulance dataset...")
ds = load_dataset('mlnomad/imnet1k_ambulance')

# Print dataset information
print("\n=== DATASET INFO ===")
print(f"Number of rows: {len(ds['train'])}")
print(f"Features: {ds['train'].features}")
print(f"Dataset size: {ds['train'].num_rows} samples")

# Print first few samples
print("\n=== SAMPLE DATA ===")
for i in range(min(3, len(ds['train']))):
    sample = ds['train'][i]
    print(f"Sample {i+1}:")
    print(f"  Label: {sample['label']}")
    if 'image' in sample:
        print(f"  Image type: {type(sample['image'])}")
        if hasattr(sample['image'], 'size'):
            print(f"  Image size: {sample['image'].size}")
    print()

# Check column names
print(f"Column names: {ds['train'].column_names}")

# Print some statistics
if 'label' in ds['train'].column_names:
    labels = ds['train']['label']
    unique_labels = set(labels)
    print(f"\nUnique labels: {unique_labels}")
    
    # Count label occurrences
    label_counts = {}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print("Label distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count} images")
