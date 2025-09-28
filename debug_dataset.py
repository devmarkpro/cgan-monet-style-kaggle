import torch
from dataset import Dataset

print("Testing dataset...")
ds = Dataset(data_dir='./data/monet/training', batch_size=128, workers=1, artifacts_folder='./artifacts')
print(f'Dataset length: {len(ds.dataset)}')
print(f'Dataloader length: {len(ds.dataloader)}')
print(f'Expected batches: {len(ds.dataset)} / {ds.batch_size} = {len(ds.dataset) / ds.batch_size}')

# Test iteration
count = 0
try:
    for i, data in enumerate(ds.dataloader):
        print(f'Batch {i}: shape {data[0].shape}')
        count += 1
        if count >= 5:  # Only show first 5 batches
            break
    print(f'Total batches processed in test: {count}')
except Exception as e:
    print(f'Error during iteration: {e}')
    import traceback
    traceback.print_exc()
