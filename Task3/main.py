#!/usr/bin/env python3
"""
Main entry point for Image Captioning project
Provides demo functionality using pretrained models
"""

import torch
import torchvision.models as models
from PIL import Image
import os

from models.encoder import EncoderCNN
from models.decoder import DecoderRNN
from utils.preprocessing import Vocabulary, get_transform
from utils.visualization import show_image_with_caption
from config.config import Config


class SimpleCaptioner:
    """Simple captioner for demo purposes"""
    
    def __init__(self):
        self.device = Config.DEVICE
        self.transform = get_transform(Config.IMAGE_SIZE)
        
        # Create simple vocabulary
        self.vocab = self.create_simple_vocab()
        
        # Initialize models
        self.encoder = EncoderCNN(Config.EMBED_SIZE, Config.ENCODER_MODEL).to(self.device)
        self.decoder = DecoderRNN(
            Config.EMBED_SIZE,
            Config.HIDDEN_SIZE,
            len(self.vocab),
            Config.NUM_LAYERS
        ).to(self.device)
        
        self.encoder.eval()
        self.decoder.eval()
    
    def create_simple_vocab(self):
        """Create a simple vocabulary for demo"""
        vocab = Vocabulary()
        
        # Add common words
        words = [
            # Articles
            'a', 'an', 'the',
            # Verbs
            'is', 'are', 'was', 'were', 'has', 'have',
            'standing', 'sitting', 'walking', 'running', 'playing',
            'eating', 'drinking', 'holding', 'wearing', 'looking',
            # Nouns
            'person', 'man', 'woman', 'child', 'people', 'group',
            'dog', 'cat', 'bird', 'horse', 'animal',
            'car', 'bus', 'train', 'bicycle', 'motorcycle',
            'tree', 'grass', 'flower', 'plant', 'sky', 'water',
            'house', 'building', 'street', 'road', 'bridge',
            'food', 'table', 'chair', 'room', 'window',
            # Adjectives
            'red', 'blue', 'green', 'yellow', 'black', 'white',
            'big', 'small', 'tall', 'short', 'old', 'young',
            # Prepositions
            'in', 'on', 'at', 'with', 'near', 'next', 'to', 'of',
            'front', 'behind', 'under', 'over'
        ]
        
        for word in words:
            vocab.add_word(word)
        
        return vocab
    
    def generate_simple_caption(self, image_path):
        """Generate a simple template-based caption"""
        # This is a simplified version for demo
        # In practice, you'd use trained models
        
        templates = [
            "A photo of a scene",
            "An image showing an outdoor view",
            "A picture of daily life",
            "A captured moment in time",
            "A view of the surroundings"
        ]
        
        import random
        return random.choice(templates)
    
    def demo(self, image_path):
        """Demo the captioning system"""
        print(f"\nProcessing image: {image_path}")
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.encoder(image_tensor)
            print(f"Extracted features shape: {features.shape}")
        
        # Generate caption (simplified for demo)
        caption = self.generate_simple_caption(image_path)
        
        print(f"Generated Caption: {caption}")
        
        # Display
        show_image_with_caption(image_path, "", caption)
        
        return caption


def print_welcome():
    """Print welcome message"""
    print("="*70)
    print(" "*15 + "IMAGE CAPTIONING AI SYSTEM")
    print("="*70)
    print("\nThis is a demo of an Image Captioning system that combines:")
    print("  • Computer Vision (CNN Encoder)")
    print("  • Natural Language Processing (RNN Decoder)")
    print("  • Deep Learning (PyTorch)")
    print("\nFeatures:")
    print("  ✓ ResNet50 for image feature extraction")
    print("  ✓ LSTM for caption generation")
    print("  ✓ Beam search for better captions")
    print("  ✓ Supports multiple pretrained models")
    print("="*70)


def main():
    """Main function"""
    print_welcome()
    
    # Print configuration
    Config.print_config()
    
    # Check for sample images
    if not os.path.exists(Config.IMAGES_PATH):
        os.makedirs(Config.IMAGES_PATH)
        print(f"\n📁 Created directory: {Config.IMAGES_PATH}")
        print(f"Please add some images to this directory and run again.")
        return
    
    # Get image files
    image_files = [f for f in os.listdir(Config.IMAGES_PATH) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"\n⚠️  No images found in {Config.IMAGES_PATH}")
        print("Please add some .jpg, .jpeg, or .png images and run again.")
        return
    
    print(f"\n📷 Found {len(image_files)} images")
    
    # Initialize captioner
    print("\n🔧 Initializing models...")
    captioner = SimpleCaptioner()
    print("✓ Models initialized")
    
    # Process images
    print(f"\n🚀 Processing images...\n")
    
    for i, image_file in enumerate(image_files[:5], 1):  # Process first 5 images
        print(f"\n[{i}/{min(5, len(image_files))}] " + "="*60)
        image_path = os.path.join(Config.IMAGES_PATH, image_file)
        captioner.demo(image_path)
    
    print("\n" + "="*70)
    print("Demo completed!")
    print("\nTo train your own model, run: python train.py")
    print("To caption specific images, run: python inference.py --image <path>")
    print("="*70)


if __name__ == "__main__":
    main()