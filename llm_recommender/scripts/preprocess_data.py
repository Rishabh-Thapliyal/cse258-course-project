"""Script to preprocess the data."""

import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data.preprocessing import DataPreprocessor
from utils.config import load_config
import pickle


def main(args):
    """Main preprocessing function."""
    # Load config
    config = load_config(args.config)
    data_config = config['data']
    
    print("=" * 60)
    print("Starting data preprocessing...")
    print("=" * 60)
    
    # Create preprocessor
    preprocessor = DataPreprocessor(
        user_sequences_path=data_config['user_sequences_path'],
        item_metadata_path=data_config.get('item_metadata_path'),
        min_sequence_length=data_config['min_sequence_length'],
        max_sequence_length=data_config['max_sequence_length'],
        train_ratio=data_config['train_ratio'],
        val_ratio=data_config['val_ratio'],
        seed=data_config['seed'],
        max_train_sequences=data_config.get('max_train_sequences')  # None = no truncation
    )
    
    # Load and preprocess data
    train_data, val_data, test_data = preprocessor.load_and_preprocess()
    
    # Save processed data
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving processed data to {output_dir}...")
    
    with open(output_dir / 'train_data.pkl', 'wb') as f:
        pickle.dump(train_data, f)
    
    with open(output_dir / 'val_data.pkl', 'wb') as f:
        pickle.dump(val_data, f)
    
    with open(output_dir / 'test_data.pkl', 'wb') as f:
        pickle.dump(test_data, f)
    
    # Save mappings
    preprocessor.save_mappings(output_dir)
    
    # Save dataset info
    info = {
        'num_users': len(preprocessor.user2id),
        'num_items': len(preprocessor.item2id) + 1,
        'num_train': len(train_data),
        'num_val': len(val_data),
        'num_test': len(test_data)
    }
    
    with open(output_dir / 'dataset_info.pkl', 'wb') as f:
        pickle.dump(info, f)
    
    print(f"\nPreprocessing complete! Data saved to {output_dir}")
    print("\nDataset info:")
    for key, value in info.items():
        print(f"  {key}: {value}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess recommendation data')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--output_dir', type=str, default='./data/processed',
                       help='Output directory for processed data')
    
    args = parser.parse_args()
    main(args)

