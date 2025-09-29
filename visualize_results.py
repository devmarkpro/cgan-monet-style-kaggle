import numpy as np
import matplotlib.pyplot as plt

def visualize_samples(samples_path, epoch_idx=-1, num_images=8):
    """
    Visualize generated samples from a saved numpy array.
    
    Args:
        samples_path: Path to the saved samples .npy file
        epoch_idx: Which epoch to visualize (-1 for last epoch)
        num_images: Number of images to display (max 16 for 4x4 grid)
    """
    # Load the samples
    sample_result = np.load(samples_path)
    
    print(f"Loaded samples shape: {sample_result.shape}")
    
    # Get samples from specified epoch
    if epoch_idx == -1:
        epoch_samples = sample_result[-1]  # Last epoch
        epoch_num = len(sample_result) - 1
    else:
        epoch_samples = sample_result[epoch_idx]
        epoch_num = epoch_idx
    
    print(f"Epoch {epoch_num} samples shape: {epoch_samples.shape}")
    
    # Determine grid size
    if num_images <= 4:
        rows, cols = 1, num_images
    elif num_images <= 8:
        rows, cols = 2, 4
    else:
        rows, cols = 4, 4
        num_images = min(num_images, 16)
    
    # Create the plot
    fig, axes = plt.subplots(figsize=(15, 10), nrows=rows, ncols=cols, 
                            sharey=True, sharex=True)
    
    # Handle case where we have only one row
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, ax in enumerate(axes[:num_images]):
        if i < len(epoch_samples):
            img = epoch_samples[i]
            
            # img is already a numpy array, shape should be (C, H, W)
            # Convert from (C, H, W) to (H, W, C) for matplotlib
            img = np.transpose(img, (1, 2, 0))
            
            # Denormalize from [-1, 1] to [0, 255]
            img = ((img + 1) * 255 / 2).astype(np.uint8)
            
            # Remove axes
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            
            # Display the image
            ax.imshow(img)
        else:
            # Hide unused subplots
            ax.axis('off')
    
    plt.suptitle(f'Generated Images - Epoch {epoch_num}')
    plt.tight_layout()
    plt.show()

def visualize_losses(losses_path):
    """
    Visualize training losses from a saved numpy array.
    
    Args:
        losses_path: Path to the saved losses .npy file
    """
    # Load the losses
    losses_history = np.load(losses_path)
    
    print(f"Loaded losses shape: {losses_history.shape}")
    
    # Convert to numpy array and transpose to get separate arrays for D and G losses
    losses = np.array(losses_history)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    plt.plot(losses[:, 0], label='Discriminator Loss', color='red', alpha=0.7)
    plt.plot(losses[:, 1], label='Generator Loss', color='blue', alpha=0.7)
    
    plt.title("Training Losses Over Time")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# Example usage:
if __name__ == "__main__":
    # Replace with your actual file paths
    samples_file = "./artifacts/samples_run_654.npy"
    losses_file = "./artifacts/losses_run_654.npy"
    
    try:
        print("Visualizing samples...")
        visualize_samples(samples_file, epoch_idx=-1, num_images=8)
        
        print("\nVisualizing losses...")
        visualize_losses(losses_file)
        
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        print("Make sure to update the file paths to match your actual saved files.")
    except Exception as e:
        print(f"Error: {e}")
