"""Simple wrapper around base tokenizer - no vocab extension."""

from typing import Dict
import torch
from transformers import AutoTokenizer


class SimpleTokenizer:
    """Simple tokenizer wrapper - keeps vocab fixed, no user/item tokens."""
    
    def __init__(self, base_tokenizer_name: str):
        """
        Args:
            base_tokenizer_name: Name of the base tokenizer (e.g., 'gpt2')
        """
        self.tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_name)
        
        # Set padding token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.vocab_size = len(self.tokenizer)
        
        print(f"Tokenizer loaded:")
        print(f"  Vocab size: {self.vocab_size}")
        print(f"  Pad token: {self.tokenizer.pad_token}")
    
    def encode(
        self,
        text: str,
        max_length: int = 512,
        truncation: bool = True,
        padding: str = 'max_length',
        return_tensors: str = 'pt'
    ) -> Dict[str, torch.Tensor]:
        """Encode text to token IDs."""
        return self.tokenizer(
            text,
            max_length=max_length,
            truncation=truncation,
            padding=padding,
            return_tensors=return_tensors
        )
    
    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def batch_encode(
        self,
        texts: list,
        max_length: int = 512,
        truncation: bool = True,
        padding: str = 'max_length',
        return_tensors: str = 'pt'
    ) -> Dict[str, torch.Tensor]:
        """Batch encode texts."""
        return self.tokenizer(
            texts,
            max_length=max_length,
            truncation=truncation,
            padding=padding,
            return_tensors=return_tensors
        )
    
    def __len__(self) -> int:
        """Return vocabulary size."""
        return self.vocab_size
    
    def save(self, save_path: str):
        """Save tokenizer."""
        self.tokenizer.save_pretrained(save_path)
    
    @classmethod
    def load(cls, load_path: str):
        """Load tokenizer."""
        return cls(load_path)
