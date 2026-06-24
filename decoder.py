"""
RNN Decoder for generating captions
Uses LSTM to generate word sequences
"""

import torch
import torch.nn as nn


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        """
        Initialize the decoder
        
        Args:
            embed_size: Dimension of word embeddings
            hidden_size: Dimension of LSTM hidden states
            vocab_size: Size of vocabulary
            num_layers: Number of LSTM layers
        """
        super(DecoderRNN, self).__init__()
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        # Word embedding layer
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            embed_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=0.5 if num_layers > 1 else 0
        )
        
        # Linear layer to convert LSTM output to vocabulary scores
        self.linear = nn.Linear(hidden_size, vocab_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, features, captions, lengths):
        """
        Forward pass during training
        
        Args:
            features: Image features from encoder (batch_size, embed_size)
            captions: Ground truth captions (batch_size, max_seq_length)
            lengths: Actual lengths of captions (batch_size,)
        
        Returns:
            outputs: Predicted word scores (batch_size, max_seq_length, vocab_size)
        """
        # Embed captions
        embeddings = self.embed(captions)
        
        # Concatenate image features with caption embeddings
        # features: (batch_size, embed_size) -> (batch_size, 1, embed_size)
        features = features.unsqueeze(1)
        embeddings = torch.cat((features, embeddings), dim=1)
        
        # Pack padded sequence
        packed = nn.utils.rnn.pack_padded_sequence(
            embeddings,
            lengths,
            batch_first=True,
            enforce_sorted=False
        )
        
        # LSTM forward pass
        hiddens, _ = self.lstm(packed)
        
        # Unpack
        hiddens, _ = nn.utils.rnn.pad_packed_sequence(hiddens, batch_first=True)
        
        # Linear layer
        outputs = self.linear(hiddens)
        
        return outputs
    
    def sample(self, features, max_length=20, start_token=1):
        """
        Generate captions using greedy search
        
        Args:
            features: Image features (batch_size, embed_size)
            max_length: Maximum length of caption
            start_token: Index of start token
        
        Returns:
            captions: Generated caption indices (batch_size, max_length)
        """
        batch_size = features.size(0)
        captions = []
        
        # Initialize LSTM state
        states = None
        
        # Start with <START> token
        inputs = features.unsqueeze(1)  # (batch_size, 1, embed_size)
        
        for i in range(max_length):
            # LSTM forward pass
            hiddens, states = self.lstm(inputs, states)
            
            # Linear layer
            outputs = self.linear(hiddens.squeeze(1))  # (batch_size, vocab_size)
            
            # Get predicted word
            _, predicted = outputs.max(1)  # (batch_size,)
            captions.append(predicted)
            
            # Prepare next input
            inputs = self.embed(predicted).unsqueeze(1)  # (batch_size, 1, embed_size)
        
        # Stack captions
        captions = torch.stack(captions, dim=1)  # (batch_size, max_length)
        
        return captions
    
    def beam_search(self, features, beam_width=3, max_length=20, start_token=1, end_token=2):
        """
        Generate captions using beam search
        
        Args:
            features: Image features (1, embed_size)
            beam_width: Width of beam
            max_length: Maximum caption length
            start_token: Start token index
            end_token: End token index
        
        Returns:
            best_caption: Best caption found
        """
        k = beam_width
        vocab_size = self.vocab_size
        
        # Initialize
        inputs = features.unsqueeze(1)  # (1, 1, embed_size)
        states = None
        
        # Start with <START> token
        hiddens, states = self.lstm(inputs, states)
        outputs = self.linear(hiddens.squeeze(1))  # (1, vocab_size)
        
        # Get top k predictions
        log_probs, indices = torch.topk(torch.log_softmax(outputs, dim=1), k)
        
        # Initialize beams
        beams = [(idx.item(), [idx.item()], log_prob.item(), states) 
                 for idx, log_prob in zip(indices[0], log_probs[0])]
        
        complete_beams = []
        
        for _ in range(max_length - 1):
            new_beams = []
            
            for word_idx, caption, log_prob, state in beams:
                if word_idx == end_token:
                    complete_beams.append((caption, log_prob))
                    continue
                
                # Embed current word
                inputs = self.embed(torch.tensor([word_idx]).to(features.device)).unsqueeze(0)
                
                # LSTM forward
                hiddens, new_state = self.lstm(inputs, state)
                outputs = self.linear(hiddens.squeeze(1))
                
                # Get top k predictions
                log_probs_step, indices_step = torch.topk(
                    torch.log_softmax(outputs, dim=1), k
                )
                
                # Add to new beams
                for idx, log_prob_step in zip(indices_step[0], log_probs_step[0]):
                    new_caption = caption + [idx.item()]
                    new_log_prob = log_prob + log_prob_step.item()
                    new_beams.append((idx.item(), new_caption, new_log_prob, new_state))
            
            # Keep top k beams
            beams = sorted(new_beams, key=lambda x: x[2], reverse=True)[:k]
            
            if not beams:
                break
        
        # Add remaining beams to complete beams
        complete_beams.extend([(caption, log_prob) for _, caption, log_prob, _ in beams])
        
        # Return best caption
        if complete_beams:
            best_caption = max(complete_beams, key=lambda x: x[1])[0]
        else:
            best_caption = beams[0][1] if beams else [start_token, end_token]
        
        return best_caption