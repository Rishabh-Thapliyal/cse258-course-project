"""Data preprocessing utilities."""

import gzip
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm


class DataPreprocessor:
    """Preprocesses user sequences and item metadata for LLM-based recommendation."""
    
    def __init__(
        self,
        user_sequences_path: str,
        item_metadata_path: Optional[str] = None,
        min_sequence_length: int = 5,
        max_sequence_length: int = 50,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
        max_train_sequences: Optional[int] = None
    ):
        self.user_sequences_path = user_sequences_path
        self.item_metadata_path = item_metadata_path
        self.min_sequence_length = min_sequence_length
        self.max_sequence_length = max_sequence_length
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1.0 - train_ratio - val_ratio
        self.seed = seed
        self.max_train_sequences = max_train_sequences
        
        np.random.seed(seed)
        
        # Mappings
        self.user2id: Dict[str, int] = {}
        self.item2id: Dict[str, int] = {}
        self.id2user: Dict[int, str] = {}
        self.id2item: Dict[int, str] = {}
        
        # Data
        self.user_sequences: Dict[int, List[Dict]] = {}
        self.item_metadata: Dict[int, Dict] = {}
        
    def load_and_preprocess(self) -> Tuple[Dict, Dict, Dict]:
        """Load and preprocess all data."""
        print("Loading user sequences...")
        self._load_user_sequences()
        
        if self.item_metadata_path:
            print("Loading item metadata...")
            self._load_item_metadata()
        
        # Truncate user sequences first if requested
        if self.max_train_sequences is not None:
            print(f"\nTruncating to {self.max_train_sequences} sequences...")
            self._truncate_user_sequences(self.max_train_sequences)
        
        print("Creating train/val/test splits...")
        train_data, val_data, test_data = self._create_splits()
        
        # Remap item IDs to be compact (1-indexed, 0 for padding)
        if self.max_train_sequences is not None:
            print("\nRemapping item IDs to compact range...")
            train_data, val_data, test_data = self._remap_item_ids(
                train_data, val_data, test_data
            )
        
        stats = self._compute_statistics()
        print(f"\nDataset Statistics:")
        print(f"  Users: {stats['num_users']}")
        print(f"  Items: {stats['num_items']}")
        print(f"  Interactions: {stats['num_interactions']}")
        print(f"  Avg sequence length: {stats['avg_seq_length']:.2f}")
        print(f"  Train sequences: {len(train_data)}")
        print(f"  Val sequences: {len(val_data)}")
        print(f"  Test sequences: {len(test_data)}")
        
        return train_data, val_data, test_data
    
    def _load_user_sequences(self):
        """Load user interaction sequences from JSONL file."""
        user_idx = 0
        item_idx = 1  # Start from 1, reserve 0 for padding
        
        with gzip.open(self.user_sequences_path, 'rt', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading sequences"):
                data = json.loads(line.strip())
                user_id_str = data['user_id']
                sequence = data['sequence']
                
                # Filter by sequence length
                if len(sequence) < self.min_sequence_length:
                    continue
                
                # Map user ID
                if user_id_str not in self.user2id:
                    self.user2id[user_id_str] = user_idx
                    self.id2user[user_idx] = user_id_str
                    user_idx += 1
                
                user_id = self.user2id[user_id_str]
                
                # Map item IDs and process sequence
                processed_sequence = []
                for interaction in sequence:
                    item_id_str = interaction['asin']
                    
                    if item_id_str not in self.item2id:
                        self.item2id[item_id_str] = item_idx
                        self.id2item[item_idx] = item_id_str
                        item_idx += 1
                    
                    item_id = self.item2id[item_id_str]
                    
                    processed_sequence.append({
                        'item_id': item_id,
                        'timestamp': interaction['ts'],
                        'rating': interaction.get('rating', 0.0)
                    })
                
                # Sort by timestamp
                processed_sequence = sorted(processed_sequence, key=lambda x: x['timestamp'])
                
                # Truncate if too long
                if len(processed_sequence) > self.max_sequence_length:
                    processed_sequence = processed_sequence[-self.max_sequence_length:]
                
                self.user_sequences[user_id] = processed_sequence
    
    def _load_item_metadata(self):
        """Load item metadata (titles, descriptions, etc.)."""
        with gzip.open(self.item_metadata_path, 'rt', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading metadata"):
                data = json.loads(line.strip())
                item_id_str = data.get('parent_asin') or data.get('asin')
                
                if item_id_str in self.item2id:
                    item_id = self.item2id[item_id_str]
                    
                    # Extract relevant metadata
                    self.item_metadata[item_id] = {
                        'title': data.get('title', ''),
                        'description': ' '.join(data.get('description', [])) if isinstance(data.get('description'), list) else data.get('description', ''),
                        'categories': data.get('categories', []),
                        'price': data.get('price', 0.0),
                        'average_rating': data.get('average_rating', 0.0)
                    }
    
    def _create_splits(self) -> Tuple[Dict, Dict, Dict]:
        """
        Create train/validation/test splits using leave-one-out strategy.
        
        Training: Uses autoregressive objective (predict each next item in sequence)
        Validation/Test: Single next-item prediction for evaluation
        """
        train_data = {}
        val_data = {}
        test_data = {}
        
        for user_id, sequence in self.user_sequences.items():
            seq_len = len(sequence)
            
            if seq_len < 3:
                continue
            
            # Leave-one-out split: last item for test, second-to-last for val
            train_seq = sequence[:-2]  # All items up to T-2
            val_item = sequence[-2]     # Item at T-1
            test_item = sequence[-1]    # Item at T
            
            if len(train_seq) >= self.min_sequence_length - 1:
                # Training: Full sequence for autoregressive training
                # Model will predict item at position i given items 0:i-1
                train_data[user_id] = {
                    'sequence': train_seq,
                    'is_autoregressive': True  # Flag for autoregressive loss
                }
                
                # Validation: Predict single next item (for evaluation metrics)
                val_data[user_id] = {
                    'sequence': train_seq,
                    'target': val_item,
                    'is_autoregressive': False
                }
                
                # Test: Predict single next item (for final evaluation)
                test_data[user_id] = {
                    'sequence': sequence[:-1],  # Include validation item
                    'target': test_item,
                    'is_autoregressive': False
                }
        
        return train_data, val_data, test_data
    
    def _truncate_user_sequences(self, max_sequences: int):
        """
        Truncate user_sequences to max_sequences before creating splits.
        This ensures val/test items will be in the training sequences.
        """
        user_ids = list(self.user_sequences.keys())
        if len(user_ids) > max_sequences:
            # Randomly sample max_sequences users
            np.random.shuffle(user_ids)
            user_ids = user_ids[:max_sequences]
            self.user_sequences = {uid: self.user_sequences[uid] for uid in user_ids}
            print(f"  Truncated to {len(self.user_sequences)} user sequences")
            
            # Update user mappings to only include kept users
            kept_user_ids = set(user_ids)
            new_user2id = {}
            new_id2user = {}
            new_user_idx = 0
            
            for old_user_id in sorted(kept_user_ids):
                old_user_str = self.id2user[old_user_id]
                new_user2id[old_user_str] = new_user_idx
                new_id2user[new_user_idx] = old_user_str
                new_user_idx += 1
            
            # Remap user IDs in sequences
            remapped_sequences = {}
            for old_user_id, sequence in self.user_sequences.items():
                old_user_str = self.id2user[old_user_id]
                new_user_id = new_user2id[old_user_str]
                remapped_sequences[new_user_id] = sequence
            
            self.user_sequences = remapped_sequences
            self.user2id = new_user2id
            self.id2user = new_id2user
    
    def _remap_item_ids(
        self,
        train_data: Dict,
        val_data: Dict,
        test_data: Dict
    ) -> Tuple[Dict, Dict, Dict]:
        """
        Remap item IDs to compact range (1-indexed, 0 reserved for padding).
        This ensures item IDs are consecutive starting from 1.
        """
        # Collect all items that appear in final dataset
        all_items = set()
        
        # From training sequences
        for user_data in train_data.values():
            for item in user_data['sequence']:
                all_items.add(item['item_id'])
        
        # From validation sequences
        for user_data in val_data.values():
            for item in user_data['sequence']:
                all_items.add(item['item_id'])
            all_items.add(user_data['target']['item_id'])
        
        # From test sequences
        for user_data in test_data.values():
            for item in user_data['sequence']:
                all_items.add(item['item_id'])
            all_items.add(user_data['target']['item_id'])
        
        print(f"  Found {len(all_items)} unique items in final dataset")
        
        # Create mapping: old_item_id -> new_item_id
        old_to_new_item = {}
        new_item_idx = 1  # Start from 1, reserve 0 for padding
        
        # Sort items for consistent mapping
        sorted_items = sorted(all_items)
        for old_item_id in sorted_items:
            old_to_new_item[old_item_id] = new_item_idx
            new_item_idx += 1
        
        # Update item2id and id2item mappings
        new_item2id = {}
        new_id2item = {}
        for old_item_id, new_item_id in old_to_new_item.items():
            old_item_str = self.id2item[old_item_id]
            new_item2id[old_item_str] = new_item_id
            new_id2item[new_item_id] = old_item_str
        
        self.item2id = new_item2id
        self.id2item = new_id2item
        
        # Remap item IDs in all data
        def remap_sequence(sequence):
            return [{
                **item,
                'item_id': old_to_new_item[item['item_id']]
            } for item in sequence]
        
        def remap_target(target):
            return {
                **target,
                'item_id': old_to_new_item[target['item_id']]
            }
        
        # Remap training data
        remapped_train_data = {}
        for user_id, user_data in train_data.items():
            remapped_train_data[user_id] = {
                'sequence': remap_sequence(user_data['sequence']),
                'is_autoregressive': user_data['is_autoregressive']
            }
        
        # Remap validation data
        remapped_val_data = {}
        for user_id, user_data in val_data.items():
            remapped_val_data[user_id] = {
                'sequence': remap_sequence(user_data['sequence']),
                'target': remap_target(user_data['target']),
                'is_autoregressive': user_data['is_autoregressive']
            }
        
        # Remap test data
        remapped_test_data = {}
        for user_id, user_data in test_data.items():
            remapped_test_data[user_id] = {
                'sequence': remap_sequence(user_data['sequence']),
                'target': remap_target(user_data['target']),
                'is_autoregressive': user_data['is_autoregressive']
            }
        
        # Update item_metadata to only include items in final dataset
        remapped_item_metadata = {}
        for old_item_id, new_item_id in old_to_new_item.items():
            if old_item_id in self.item_metadata:
                remapped_item_metadata[new_item_id] = self.item_metadata[old_item_id]
        
        self.item_metadata = remapped_item_metadata
        
        print(f"  Remapped {len(old_to_new_item)} items to new IDs (1-{len(old_to_new_item)})")
        print(f"  Updated item metadata: {len(self.item_metadata)} items")
        
        return remapped_train_data, remapped_val_data, remapped_test_data
    
    def _compute_statistics(self) -> Dict:
        """Compute dataset statistics."""
        num_users = len(self.user2id)
        num_items = len(self.item2id)
        num_interactions = sum(len(seq) for seq in self.user_sequences.values())
        avg_seq_length = num_interactions / num_users if num_users > 0 else 0
        
        return {
            'num_users': num_users,
            'num_items': num_items,
            'num_interactions': num_interactions,
            'avg_seq_length': avg_seq_length
        }
    
    def save_mappings(self, save_dir: str):
        """Save user/item mappings to disk."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        with open(save_path / 'user2id.pkl', 'wb') as f:
            pickle.dump(self.user2id, f)
        
        with open(save_path / 'item2id.pkl', 'wb') as f:
            pickle.dump(self.item2id, f)
        
        with open(save_path / 'id2user.pkl', 'wb') as f:
            pickle.dump(self.id2user, f)
        
        with open(save_path / 'id2item.pkl', 'wb') as f:
            pickle.dump(self.id2item, f)
        
        with open(save_path / 'item_metadata.pkl', 'wb') as f:
            pickle.dump(self.item_metadata, f)
        
        print(f"Mappings saved to {save_path}")
    
    @staticmethod
    def load_mappings(load_dir: str) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """Load user/item mappings from disk."""
        load_path = Path(load_dir)
        
        with open(load_path / 'user2id.pkl', 'rb') as f:
            user2id = pickle.load(f)
        
        with open(load_path / 'item2id.pkl', 'rb') as f:
            item2id = pickle.load(f)
        
        with open(load_path / 'id2user.pkl', 'rb') as f:
            id2user = pickle.load(f)
        
        with open(load_path / 'id2item.pkl', 'rb') as f:
            id2item = pickle.load(f)
        
        with open(load_path / 'item_metadata.pkl', 'rb') as f:
            item_metadata = pickle.load(f)
        
        return user2id, item2id, id2user, id2item, item_metadata

