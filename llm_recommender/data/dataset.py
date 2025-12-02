"""PyTorch dataset and data module for recommendation."""

import random
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm


class RecDataset(Dataset):
    """Dataset for LLM-based recommendation."""
    
    def __init__(
        self,
        data: Dict[int, Dict],
        item_metadata: Dict[int, Dict],
        num_items: int,
        mode: str = 'train',
        negative_samples: int = 5,
        max_seq_length: int = 50,
        eval_negatives: int = 100,
        eval_seed: int = 42
    ):
        """
        Args:
            data: Dictionary mapping user_id to {sequence, target}
            item_metadata: Dictionary mapping item_id to metadata
            num_items: Total number of items
            mode: 'train', 'val', or 'test'
            negative_samples: Number of negative samples for training
            max_seq_length: Maximum sequence length
            eval_negatives: Number of negative samples for evaluation (val/test) - SASRec uses 100
            eval_seed: Random seed for precomputing evaluation negative samples (default: 42)
        """
        self.data = data
        self.item_metadata = item_metadata
        self.num_items = num_items
        self.mode = mode
        self.negative_samples = negative_samples
        self.max_seq_length = max_seq_length
        self.eval_negatives = eval_negatives
        self.eval_seed = eval_seed
        
        self.user_ids = list(data.keys())
        
        # Precompute negative samples for val/test (SASRec protocol: 100 negatives per user)
        if self.mode in ['val', 'test']:
            self.eval_negative_samples = {}
            np.random.seed(eval_seed)  # Fixed seed for reproducibility
            
            # Precompute all valid item IDs once (exclude padding 0)
            all_items = np.arange(1, self.num_items, dtype=np.int32)  # (num_items-1,)
            
            # Pre-allocate boolean mask for faster operations
            item_mask = np.ones(self.num_items, dtype=bool)  # True = available as negative
            item_mask[0] = False  # Exclude padding (item_id=0)
            
            for user_id in tqdm(self.user_ids, desc="Precomputing negative samples"):
                user_data = self.data[user_id]
                sequence = user_data['sequence']
                target = user_data.get('target', {})
                
                # Get all items user has interacted with (sequence + target)
                user_items_list = [item['item_id'] for item in sequence]
                if target and 'item_id' in target:
                    user_items_list.append(target['item_id'])
                
                # Reset mask (set all to True except padding)
                item_mask.fill(True)
                item_mask[0] = False
                
                # Mark user's items as False (not available as negatives)
                if user_items_list:
                    user_items_arr = np.array(user_items_list, dtype=np.int32)
                    # Only mark items that are within valid range
                    valid_mask = (user_items_arr > 0) & (user_items_arr < self.num_items)
                    item_mask[user_items_arr[valid_mask]] = False
                
                # Get negative candidates using boolean indexing (much faster)
                neg_candidates = all_items[item_mask[1:]]  # Skip padding at index 0
                
                # Sample exactly eval_negatives (100) negatives
                if len(neg_candidates) >= self.eval_negatives:
                    self.eval_negative_samples[user_id] = np.random.choice(
                        neg_candidates, 
                        size=self.eval_negatives, 
                        replace=False
                    ).tolist()
                else:
                    # If not enough candidates, use all available and pad with replacement
                    self.eval_negative_samples[user_id] = (
                        neg_candidates.tolist() + 
                        np.random.choice(
                            neg_candidates,
                            size=self.eval_negatives - len(neg_candidates),
                            replace=True
                        ).tolist()
                    )
        
    def __len__(self) -> int:
        return len(self.user_ids)
    
    def __getitem__(self, idx: int) -> Dict:
        user_id = self.user_ids[idx]
        user_data = self.data[user_id]
        
        sequence = user_data['sequence']
        is_autoregressive = user_data.get('is_autoregressive', False)
        
        # Extract item IDs from sequence
        item_ids = [item['item_id'] for item in sequence]
        ratings = [item['rating'] for item in sequence]
        
        # Pad or truncate sequence
        if len(item_ids) > self.max_seq_length:
            item_ids = item_ids[-self.max_seq_length:]
            ratings = ratings[-self.max_seq_length:]
        
        seq_length = len(item_ids)
        
        # Handle target based on mode
        if is_autoregressive:
            # Training: autoregressive targets (each next item in sequence)
            # Targets will be item_ids[1:] (shifted by 1)
            target_item_id = None  # Will create targets in collate_fn
            target_item_ids = item_ids[1:] + [0]  # Shift left, pad last position
        else:
            # Validation/Test: single next-item prediction
            target = user_data['target']
            target_item_id = target['item_id']
            target_item_ids = None
        
        # Sample negative items
        negative_items = []
        if self.mode == 'train':
            # Training: Sample negatives for BPR loss
            if is_autoregressive:
                # For autoregressive: exclude all items in the sequence
                user_items = set(item_ids)
            else:
                # For single-target: exclude sequence items and target
                user_items = set(item_ids + [target_item_id])
            
            # Exclude padding (0) and user's items
            neg_candidates = list(set(range(1, self.num_items)) - user_items)
            
            if len(neg_candidates) >= self.negative_samples:
                negative_items = random.sample(neg_candidates, self.negative_samples)
            else:
                negative_items = neg_candidates + random.choices(
                    neg_candidates, 
                    k=self.negative_samples - len(neg_candidates)
                )
        elif self.mode in ['val', 'test']:
            # Validation/Test: Use precomputed 100 negatives (SASRec protocol)
            negative_items = self.eval_negative_samples[user_id]
        
        # Get item content (titles, descriptions, categories)
        # Categories will be embedded as text tokens and concatenated with other text tokens
        item_texts = []
        item_categories = []  # Store categories separately for explicit tokenization
        for item_id in item_ids:
            if item_id in self.item_metadata:
                metadata = self.item_metadata[item_id]
                # Format categories as text
                categories = metadata.get('categories', [])
                category_text = ""
                if categories:
                    if isinstance(categories, list):
                        category_text = "Categories: " + ", ".join(str(c) for c in categories if c)
                    else:
                        category_text = f"Categories: {categories}"
                item_categories.append(category_text)
                # Combine: categories + title + description (categories will be tokenized first)
                text = f"{category_text} {metadata.get('title', '')} {metadata.get('description', '')}"
                item_texts.append(text.strip())
            else:
                item_texts.append("")
                item_categories.append("")
        
        target_text = ""
        target_categories = ""
        if target_item_id is not None and target_item_id in self.item_metadata:
            metadata = self.item_metadata[target_item_id]
            # Format categories as text
            categories = metadata.get('categories', [])
            if categories:
                if isinstance(categories, list):
                    target_categories = "Categories: " + ", ".join(str(c) for c in categories if c)
                else:
                    target_categories = f"Categories: {categories}"
            # Combine: categories + title + description
            target_text = f"{target_categories} {metadata.get('title', '')} {metadata.get('description', '')}"
        
        return {
            'user_id': user_id,
            'item_ids': item_ids,
            'ratings': ratings,
            'seq_length': seq_length,
            'target_item_id': target_item_id,  # None for autoregressive
            'target_item_ids': target_item_ids,  # Sequence of targets for autoregressive
            'negative_items': negative_items,
            'item_texts': item_texts,
            'item_categories': item_categories,  # Categories for explicit tokenization
            'target_text': target_text,
            'target_categories': target_categories,
            'is_autoregressive': is_autoregressive
        }


class RecDataModule:
    """Data module for handling train/val/test data loaders."""
    
    def __init__(
        self,
        train_data: Dict,
        val_data: Dict,
        test_data: Dict,
        item_metadata: Dict,
        num_items: int,
        tokenizer = None,
        batch_size: int = 32,
        num_workers: int = 4,
        negative_samples: int = 5,
        max_seq_length: int = 50,
        max_text_length: int = 128,
        eval_seed: int = 42
    ):
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.item_metadata = item_metadata
        self.num_items = num_items
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.negative_samples = negative_samples
        self.max_seq_length = max_seq_length
        self.max_text_length = max_text_length
        self.eval_seed = eval_seed
        
        # Create datasets
        self.train_dataset = RecDataset(
            train_data, item_metadata, num_items, 
            mode='train', 
            negative_samples=negative_samples,
            max_seq_length=max_seq_length,
            eval_seed=eval_seed
        )
        
        self.val_dataset = RecDataset(
            val_data, item_metadata, num_items, 
            mode='val',
            max_seq_length=max_seq_length,
            eval_seed=eval_seed
        )
        
        self.test_dataset = RecDataset(
            test_data, item_metadata, num_items, 
            mode='test',
            max_seq_length=max_seq_length,
            eval_seed=eval_seed
        )
    
    def collate_fn(self, batch: List[Dict]) -> Dict:
        """Collate function for batching."""
        user_ids = [item['user_id'] for item in batch]
        seq_lengths = [item['seq_length'] for item in batch]
        is_autoregressive = batch[0]['is_autoregressive']
        
        # Pad sequences
        max_len = max(seq_lengths)
        
        item_ids_padded = []
        ratings_padded = []
        attention_mask = []
        target_item_ids_padded = []  # For autoregressive targets
        
        for item in batch:
            item_ids = item['item_ids']
            ratings = item['ratings']
            pad_len = max_len - len(item_ids)
            
            item_ids_padded.append(item_ids + [0] * pad_len)
            ratings_padded.append(ratings + [0.0] * pad_len)
            attention_mask.append([1] * len(item_ids) + [0] * pad_len)
            
            # For autoregressive: pad target sequence
            if is_autoregressive:
                target_ids = item['target_item_ids']
                target_item_ids_padded.append(target_ids + [0] * pad_len)
        
        # Handle targets based on mode
        if is_autoregressive:
            # Autoregressive: targets are shifted input sequence
            target_item_ids = torch.LongTensor(target_item_ids_padded)
        else:
            # Single-target: collect single targets
            target_item_ids = torch.LongTensor([item['target_item_id'] for item in batch])
        
        # Negative items for BPR loss (training) or evaluation (val/test)
        negative_items = []
        if batch[0]['negative_items']:
            negative_items = [item['negative_items'] for item in batch]
            # For val/test, negative_items should be a list of 100 negatives per user
            # For train, negative_items is a list of negative_samples per user
        
        # Item texts
        item_texts = [item['item_texts'] for item in batch]
        target_texts = [item['target_text'] for item in batch]
        
        # Tokenize item texts for content loss (if tokenizer available)
        # Categories are embedded as text tokens and concatenated with title/description tokens
        text_input_ids = None
        text_attention_mask = None
        text_labels = None
        content_item_ids = None
        
        if self.tokenizer is not None:
            # Collect texts and categories separately for explicit concatenation
            texts_to_tokenize = []
            valid_item_ids = []
            
            if is_autoregressive:
                # In autoregressive mode, use first item from each sequence for content loss
                for item in batch:
                    if len(item['item_texts']) > 0:
                        text = item['item_texts'][0]  # Already includes categories
                        if text.strip():
                            texts_to_tokenize.append(text)
                            valid_item_ids.append(item['item_ids'][0])
            else:
                # In single-target mode, use target item text
                for idx, text in enumerate(target_texts):
                    if text.strip():  # Already includes categories
                        texts_to_tokenize.append(text)
                        valid_item_ids.append(batch[idx]['target_item_id'])
            
            if texts_to_tokenize:
                # Tokenize all texts at once (categories are already included in the text)
                # The tokenizer will handle: "Categories: cat1, cat2 Title Description"
                tokenized = self.tokenizer(
                    texts_to_tokenize,
                    max_length=self.max_text_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                text_input_ids = tokenized['input_ids']
                text_attention_mask = tokenized['attention_mask']
                # Labels for LM loss (same as input_ids, will shift inside model)
                text_labels = text_input_ids.clone()
                content_item_ids = torch.LongTensor(valid_item_ids)
        
        return {
            'user_ids': torch.LongTensor(user_ids),
            'item_ids': torch.LongTensor(item_ids_padded),
            'ratings': torch.FloatTensor(ratings_padded),
            'attention_mask': torch.LongTensor(attention_mask),
            'target_item_ids': target_item_ids,  # (B,) for single-target, (B, L) for autoregressive
            'negative_items': torch.LongTensor(negative_items) if negative_items else None,
            'seq_lengths': torch.LongTensor(seq_lengths),
            'item_texts': item_texts,
            'target_texts': target_texts,
            'is_autoregressive': is_autoregressive,
            # Content loss fields
            'text_input_ids': text_input_ids,
            'text_attention_mask': text_attention_mask,
            'text_labels': text_labels,
            'content_item_ids': content_item_ids
        }
    
    def train_dataloader(self) -> DataLoader:
        """Create training data loader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation data loader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True
        )
    
    def test_dataloader(self) -> DataLoader:
        """Create test data loader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True
        )

