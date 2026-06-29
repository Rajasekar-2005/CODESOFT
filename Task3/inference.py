"""
Inference script for generating captions for new images
"""

import torch
from PIL import Image
import os
import argparse

from models.encoder import EncoderCNN
from models.decoder import DecoderRNN
from utils.preprocessing import Vocabulary, get_transform, load_image
from utils.visualization import show_image_with_caption
from config.config import Config


class ImageCaptioner:
    def __init__(self, encoder_path=None, decoder_path=None, vocab_path=None):
        """
        Initialize Image Captioner
        
        Args:
            encoder_path: Path to saved encoder model
            decoder_path: Path to saved decoder model
            vocab_path: Path to saved vocabulary
        """
        self.device = Config.DEVICE
        self.transform = get_transform(Config.IMAGE_SIZE, is_training=False)
        
        # Initialize vocabulary
        self.vocab = self.load_or_create_vocab(vocab_path)
        
        # Initialize models
        self.encoder = EncoderCNN(
            Config.EMBED_SIZE,
            Config.ENCODER_MODEL
        ).to(self.device)
        
        self.decoder = DecoderRNN(
            Config.EMBED_SIZE,
            Config.HIDDEN_SIZE,
            len(self.vocab),
            Config.NUM_LAYERS
        ).to(self.device)
        
        # Load pretrained weights if available
        if encoder_path and os.path.exists(encoder_path):
            self.encoder.load_state_dict(torch.load(encoder_path, map_location=self.device))
            print(f"Loaded encoder from {encoder_path}")
        
        if decoder_path and os.path.exists(decoder_path):
            self.decoder.load_state_dict(torch.load(decoder_path, map_location=self.device))
            print(f"Loaded decoder from {decoder_path}")
        
        # Set to evaluation mode
        self.encoder.eval()
        self.decoder.eval()
    
    def load_or_create_vocab(self, vocab_path):
        """Load vocabulary or create a default one"""
        if vocab_path and os.path.exists(vocab_path):
            vocab = torch.load(vocab_path)
            print(f"Loaded vocabulary from {vocab_path}")
        else:
            # Create default vocabulary with common words
            vocab = Vocabulary()
            # Add some common words (in practice, this should be loaded from training)
            common_words = [
                'a', 'an', 'the', 'is', 'are', 'was', 'were',
                'person', 'man', 'woman', 'child', 'people',
                'dog', 'cat', 'bird', 'animal',
                'car', 'bus', 'train', 'bike',
                'tree', 'grass', 'flower', 'plant',
                'house', 'building', 'street', 'road',
                'standing', 'sitting', 'walking', 'running',
                'red', 'blue', 'green', 'yellow', 'black', 'white',
                'big', 'small', 'tall', 'short',
                'in', 'on', 'at', 'with', 'near', 'next', 'to',
                'playing', 'holding', 'wearing', 'eating'
            ]
            for word in common_words:
                vocab.add_word(word)
            print("Created default vocabulary")
        
        return vocab
    
    def generate_caption(self, image_path, use_beam_search=True, beam_width=3):
        """
        Generate caption for an image
        
        Args:
            image_path: Path to image
            use_beam_search: Whether to use beam search
            beam_width: Beam width for beam search
        
        Returns:
            caption: Generated caption string
        """
        # Load and transform image
        image = load_image(image_path, self.transform)
        image = image.unsqueeze(0).to(self.device)  # Add batch dimension
        
        # Generate caption
        with torch.no_grad():
            # Extract features
            features = self.encoder(image)
            
            # Generate caption
            if use_beam_search:
                caption_indices = self.decoder.beam_search(
                    features,
                    beam_width=beam_width,
                    max_length=Config.MAX_SEQ_LENGTH,
                    start_token=self.vocab.word2idx['<START>'],
                    end_token=self.vocab.word2idx['<END>']
                )
            else:
                caption_indices = self.decoder.sample(
                    features,
                    max_length=Config.MAX_SEQ_LENGTH,
                    start_token=self.vocab.word2idx['<START>']
                )
                caption_indices = caption_indices[0].tolist()
            
            # Decode caption
            caption = self.vocab.decode(caption_indices)
        
        return caption
    
    def caption_multiple_images(self, image_dir, num_images=5):
        """
        Generate captions for multiple images
        
        Args:
            image_dir: Directory containing images
            num_images: Number of images to caption
        """
        # Get image files
        image_files = [f for f in os.listdir(image_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        image_files = image_files[:num_images]
        
        print(f"\nGenerating captions for {len(image_files)} images...\n")
        
        for image_file in image_files:
            image_path = os.path.join(image_dir, image_file)
            caption = self.generate_caption(image_path)
            
            print(f"Image: {image_file}")
            print(f"Caption: {caption}")
            print("-" * 60)
            
            # Display image with caption
            show_image_with_caption(image_path, "", caption)


def main():
    parser = argparse.ArgumentParser(description='Generate captions for images')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--image_dir', type=str, default='sample_images/', 
                       help='Directory containing images')
    parser.add_argument('--encoder', type=str, help='Path to encoder model')
    parser.add_argument('--decoder', type=str, help='Path to decoder model')
    parser.add_argument('--vocab', type=str, help='Path to vocabulary')
    parser.add_argument('--beam_width', type=int, default=3, help='Beam width for beam search')
    parser.add_argument('--num_images', type=int, default=5, help='Number of images to caption')
    
    args = parser.parse_args()
    
    # Print configuration
    Config.print_config()
    
    # Initialize captioner
    captioner = ImageCaptioner(
        encoder_path=args.encoder,
        decoder_path=args.decoder,
        vocab_path=args.vocab
    )
    
    # Generate captions
    if args.image:
        print(f"\nGenerating caption for {args.image}...\n")
        caption = captioner.generate_caption(args.image, beam_width=args.beam_width)
        print(f"Caption: {caption}\n")
        show_image_with_caption(args.image, "", caption)
    else:
        captioner.caption_multiple_images(args.image_dir, args.num_images)


if __name__ == "__main__":
    main()