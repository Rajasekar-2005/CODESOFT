"""
Training script for Image Captioning model
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm

from models.encoder import EncoderCNN
from models.decoder import DecoderRNN
from utils.preprocessing import Vocabulary, get_transform
from config.config import Config


class CaptionDataset(Dataset):
    """Dataset class for image-caption pairs"""
    
    def __init__(self, image_paths, captions, vocab, transform):
        """
        Initialize dataset
        
        Args:
            image_paths: List of image paths
            captions: List of captions
            vocab: Vocabulary object
            transform: Image transformations
        """
        self.image_paths = image_paths
        self.captions = captions
        self.vocab = vocab
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        from PIL import Image
        
        # Load image
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Encode caption
        caption = self.captions[idx]
        caption_indices = self.vocab.encode(caption)
        caption_tensor = torch.tensor(caption_indices)
        
        return image, caption_tensor


def collate_fn(data):
    """
    Custom collate function for DataLoader
    Handles variable length captions
    """
    # Sort by caption length (descending)
    data.sort(key=lambda x: len(x[1]), reverse=True)
    
    images, captions = zip(*data)
    
    # Stack images
    images = torch.stack(images, 0)
    
    # Get caption lengths
    lengths = [len(cap) for cap in captions]
    
    # Pad captions
    max_length = max(lengths)
    padded_captions = torch.zeros(len(captions), max_length).long()
    
    for i, cap in enumerate(captions):
        end = lengths[i]
        padded_captions[i, :end] = cap[:end]
    
    return images, padded_captions, lengths


class Trainer:
    """Trainer class for Image Captioning model"""
    
    def __init__(self, train_loader, vocab, device):
        """
        Initialize trainer
        
        Args:
            train_loader: DataLoader for training data
            vocab: Vocabulary object
            device: Device to train on
        """
        self.train_loader = train_loader
        self.vocab = vocab
        self.device = device
        
        # Initialize models
        self.encoder = EncoderCNN(
            Config.EMBED_SIZE,
            Config.ENCODER_MODEL
        ).to(device)
        
        self.decoder = DecoderRNN(
            Config.EMBED_SIZE,
            Config.HIDDEN_SIZE,
            len(vocab),
            Config.NUM_LAYERS
        ).to(device)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        
        params = list(self.decoder.parameters()) + list(self.encoder.linear.parameters())
        self.optimizer = torch.optim.Adam(params, lr=Config.LEARNING_RATE)
        
        # Create save directory
        os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.encoder.train()
        self.decoder.train()
        
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for images, captions, lengths in pbar:
            images = images.to(self.device)
            captions = captions.to(self.device)
            
            # Remove <END> token from targets
            targets = captions[:, 1:]
            
            # Adjust lengths
            lengths = [l - 1 for l in lengths]
            
            # Forward pass
            features = self.encoder(images)
            outputs = self.decoder(features, captions[:, :-1], lengths)
            
            # Calculate loss
            loss = self.criterion(
                outputs.reshape(-1, len(self.vocab)),
                targets.reshape(-1)
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def train(self, num_epochs):
        """Train the model"""
        print(f"\nTraining for {num_epochs} epochs...")
        
        for epoch in range(1, num_epochs + 1):
            avg_loss = self.train_epoch(epoch)
            print(f'Epoch [{epoch}/{num_epochs}], Average Loss: {avg_loss:.4f}')
            
            # Save checkpoint
            if epoch % 5 == 0:
                self.save_checkpoint(epoch)
        
        # Save final model
        self.save_checkpoint('final')
    
    def save_checkpoint(self, epoch):
        """Save model checkpoint"""
        encoder_path = os.path.join(Config.MODEL_SAVE_PATH, f'encoder_{epoch}.pth')
        decoder_path = os.path.join(Config.MODEL_SAVE_PATH, f'decoder_{epoch}.pth')
        vocab_path = os.path.join(Config.MODEL_SAVE_PATH, 'vocab.pth')
        
        torch.save(self.encoder.state_dict(), encoder_path)
        torch.save(self.decoder.state_dict(), decoder_path)
        torch.save(self.vocab, vocab_path)
        
        print(f'Saved checkpoint at epoch {epoch}')


def create_sample_data():
    """Create sample data for demonstration"""
    print("Creating sample training data...")
    
    # Sample captions (in practice, you'd load from a dataset)
    sample_data = [
        ("sample_images/image1.jpg", "a person standing in a park"),
        ("sample_images/image2.jpg", "a dog playing with a ball"),
        ("sample_images/image3.jpg", "a cat sitting on a chair"),
    ]
    
    image_paths = [path for path, _ in sample_data]
    captions = [cap for _, cap in sample_data]
    
    return image_paths, captions


def main():
    """Main training function"""
    print("="*70)
    print(" "*20 + "TRAINING IMAGE CAPTIONING MODEL")
    print("="*70)
    
    Config.print_config()
    
    # Create sample data
    image_paths, captions = create_sample_data()
    
    # Create vocabulary
    print("\nBuilding vocabulary...")
    vocab = Vocabulary()
    vocab.build_vocabulary(captions)
    print(f"Vocabulary size: {len(vocab)}")
    
    # Create dataset and dataloader
    print("\nCreating dataset...")
    transform = get_transform(Config.IMAGE_SIZE, is_training=True)
    dataset = CaptionDataset(image_paths, captions, vocab, transform)
    
    train_loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Initialize trainer
    trainer = Trainer(train_loader, vocab, Config.DEVICE)
    
    # Train model
    trainer.train(Config.NUM_EPOCHS)
    
    print("\n" + "="*70)
    print("Training completed!")
    print(f"Models saved in: {Config.MODEL_SAVE_PATH}")
    print("="*70)


if __name__ == "__main__":
    main()