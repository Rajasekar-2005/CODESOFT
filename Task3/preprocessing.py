"""
Preprocessing utilities for images and captions
"""

import torch
from torchvision import transforms
from PIL import Image
import nltk
from collections import Counter

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class Vocabulary:
    """Vocabulary class for managing word-to-index mappings"""
    
    def __init__(self, freq_threshold=5):
        """
        Initialize vocabulary
        
        Args:
            freq_threshold: Minimum frequency for a word to be included
        """
        self.freq_threshold = freq_threshold
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        
        # Add special tokens
        self.add_word('<PAD>')
        self.add_word('<START>')
        self.add_word('<END>')
        self.add_word('<UNK>')
    
    def add_word(self, word):
        """Add word to vocabulary"""
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
    
    def build_vocabulary(self, captions):
        """
        Build vocabulary from captions
        
        Args:
            captions: List of caption strings
        """
        counter = Counter()
        
        for caption in captions:
            tokens = self.tokenize(caption)
            counter.update(tokens)
        
        # Add words that meet frequency threshold
        for word, count in counter.items():
            if count >= self.freq_threshold:
                self.add_word(word)
    
    def tokenize(self, text):
        """Tokenize text"""
        return nltk.tokenize.word_tokenize(text.lower())
    
    def encode(self, caption):
        """
        Convert caption to indices
        
        Args:
            caption: Caption string
        
        Returns:
            indices: List of word indices
        """
        tokens = self.tokenize(caption)
        indices = [self.word2idx.get('<START>')]
        
        for token in tokens:
            if token in self.word2idx:
                indices.append(self.word2idx[token])
            else:
                indices.append(self.word2idx['<UNK>'])
        
        indices.append(self.word2idx['<END>'])
        
        return indices
    
    def decode(self, indices, skip_special_tokens=True):
        """
        Convert indices to caption
        
        Args:
            indices: List of word indices
            skip_special_tokens: Whether to skip special tokens
        
        Returns:
            caption: Caption string
        """
        special_tokens = ['<PAD>', '<START>', '<END>', '<UNK>']
        words = []
        
        for idx in indices:
            if isinstance(idx, torch.Tensor):
                idx = idx.item()
            
            word = self.idx2word.get(idx, '<UNK>')
            
            if skip_special_tokens and word in special_tokens:
                continue
            
            if word == '<END>':
                break
            
            words.append(word)
        
        return ' '.join(words)
    
    def __len__(self):
        """Return vocabulary size"""
        return len(self.word2idx)


def get_transform(image_size=224, is_training=False):
    """
    Get image transformation pipeline
    
    Args:
        image_size: Size to resize images
        is_training: Whether transforming for training (applies augmentation)
    
    Returns:
        transform: Transformation pipeline
    """
    if is_training:
        transform = transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    return transform


def load_image(image_path, transform=None):
    """
    Load and transform image
    
    Args:
        image_path: Path to image
        transform: Transformation to apply
    
    Returns:
        image: Transformed image tensor
    """
    image = Image.open(image_path).convert('RGB')
    
    if transform is not None:
        image = transform(image)
    
    return image