import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import NoiseReductionDataset
from model import AudioDenoiserCNN

def train_model():
    # Hyperparameters
    batch_size = 8
    learning_rate = 0.001
    epochs = 5  # Adjust epochs as needed for your training session
    sample_rate = 16000
    segment_length = 32000

    # Directories
    clean_dir = "data/clean"
    noise_dir = "data/noise"
    os.makedirs("models", exist_ok=True)

    # Check data directories
    if not os.path.exists(clean_dir) or not os.path.exists(noise_dir) or \
       len(os.listdir(clean_dir)) == 0 or len(os.listdir(noise_dir)) == 0:
        print("[ERROR] Directories 'data/clean' or 'data/noise' are empty!")
        print("Please place at least one .wav file in both folders before running training.")
        return

    print("Initializing dataset and dataloader...")
    dataset = NoiseReductionDataset(clean_dir, noise_dir, sample_rate, segment_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Device configuration (CPU optimized for your current setup)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using training device: {device}")

    # Model, Loss Function, and Optimizer
    model = AudioDenoiserCNN().to(device)
    criterion = nn.L1Loss()  # Waveform reconstruction loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("Starting training loop...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for i, (noisy, clean) in enumerate(dataloader):
            noisy = noisy.to(device)
            clean = clean.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            output = model(noisy)

            # Compute loss
            loss = criterion(output, clean)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / max(len(dataloader), 1)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    # Save trained model weights
    model_path = "models/denoiser_weights.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Training complete! Model weights saved successfully to {model_path}")

if __name__ == "__main__":
    train_model()