"""Script for inference/prediction with Stage A model."""

import sys
import argparse
import pickle
import json
import random
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import torch
from torch.amp import autocast
from tqdm import tqdm

from models import StageAModel
from data import RecDataModule
from utils.config import load_config
from utils.metrics import compute_sampled_metrics, print_metrics


def load_model(checkpoint_path, config, num_users, num_items):
    """Load trained Stage A model from checkpoint."""
    # Create model
    model = StageAModel(
        base_llm_name=config['model']['base_llm'],
        num_users=num_users,
        num_items=num_items,
        embedding_dim=config['model']['embedding_dim'],
        lambda_c=config['embeddings']['lambda_c'],
        freeze_llm=config['model']['freeze_llm_stage_a'],
        use_bpr_loss=config['stage_a']['collaborative'].get('use_bpr_loss', True),
        random_init_llm=config['model'].get('random_init_stage_a_llm', False)
    )
    
    # Load checkpoint
    checkpoint = torch.load(
        Path(checkpoint_path + '/best_model') / 'pytorch_model.pt',
        map_location='cpu',
        weights_only=False
    )
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    return model


@torch.no_grad()
def evaluate_on_test_data(model, test_loader, device='cuda', max_batches=None):
    """
    Evaluate Stage A model on test data using SASRec-style sampled evaluation.
    
    Args:
        model: Trained StageAModel
        test_loader: DataLoader for test data
        device: Device to use
        max_batches: Maximum number of batches to evaluate (None for all)
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.to(device)
    model.eval()
    
    # Setup mixed precision if needed
    use_amp = device != 'cpu'
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    amp_dtype = torch.float16 if use_amp else torch.float32
    
    all_scores = []
    all_target_indices = []
    
    for i, batch in enumerate(tqdm(test_loader, desc="Evaluating on test data")):
        if max_batches and i >= max_batches:
            break
            
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Get positive items and negative items
        pos_items = batch['target_item_ids']  # (B,)
        neg_items = batch['negative_items']    # (B, 100) - should be 100 negatives per user
        
        with autocast(device_type=device_type, enabled=use_amp, dtype=amp_dtype):
            # 1) Get LLM output
            llm_output = model.forward_collaborative(
                user_ids=batch['user_ids'],
                item_ids=batch['item_ids'],
                attention_mask=batch['attention_mask']
            )
            
            # 2) Build candidate set: [pos] + 100 negatives (shape = (B, 101))
            candidates = torch.cat([
                pos_items.unsqueeze(1),  # (B, 1)
                neg_items                # (B, 100)
            ], dim=1)  # (B, 101)
            
            # 3) Score only the candidate items (more efficient than scoring all items)
            candidate_scores = model.collab_scoring_head(
                llm_output, 
                candidate_items=candidates,
                use_fusion=True
            )  # (B, 101)
            
            # 4) Target index is always 0 in each row (positive is first in candidate list)
            batch_target_idx = torch.zeros(
                candidate_scores.size(0),
                dtype=torch.long,
                device=candidate_scores.device
            )
            
            # Accumulate
            all_scores.append(candidate_scores.cpu())
            all_target_indices.append(batch_target_idx.cpu())
    
    if not all_scores:
        return {}
    
    # Concatenate all batches
    all_scores = torch.cat(all_scores, dim=0)  # (N_users, 101)
    all_target_indices = torch.cat(all_target_indices, dim=0)  # (N_users,)
    
    # Compute SASRec-style sampled metrics
    metrics = compute_sampled_metrics(all_scores, all_target_indices, k_list=[1, 5, 10, 20])
    
    return metrics


def load_config_from_checkpoint(checkpoint_path):
    """
    Load config from checkpoint directory.
    Tries config.json first (saved during training), then falls back to config.yaml.
    """
    checkpoint_dir = Path(checkpoint_path)
    
    # Try config.json first (saved during training)
    config_json_path = checkpoint_dir / 'config.json'
    if config_json_path.exists():
        print(f"Loading config from {config_json_path}...")
        with open(config_json_path, 'r') as f:
            config = json.load(f)
        return config
    
    # Fall back to config.yaml if it exists
    config_yaml_path = checkpoint_dir / 'config.yaml'
    if config_yaml_path.exists():
        print(f"Loading config from {config_yaml_path}...")
        return load_config(str(config_yaml_path))
    
    return None


def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    """Main evaluation function."""
    # Set random seed for reproducibility (ensures same negative samples)
    eval_seed = args.seed if args.seed is not None else 42
    set_seed(eval_seed)
    print(f"Using random seed: {eval_seed} (ensures deterministic negative samples)")
    
    # Load config from checkpoint directory first
    checkpoint_path = Path(args.checkpoint)
    config = load_config_from_checkpoint(checkpoint_path)
    
    # Fall back to provided config path if not found in checkpoint
    if config is None:
        if args.config:
            print(f"Config not found in checkpoint, loading from {args.config}...")
            config = load_config(args.config)
        else:
            raise ValueError(
                f"Config not found in checkpoint directory {checkpoint_path} "
                "and no --config provided. Please provide --config or ensure "
                "config.json exists in the checkpoint directory."
            )
    
    # Load dataset info
    data_dir = Path(args.data_dir)
    
    print(f"Loading dataset info from {data_dir}...")
    with open(data_dir / 'dataset_info.pkl', 'rb') as f:
        dataset_info = pickle.load(f)
    
    with open(data_dir / 'test_data.pkl', 'rb') as f:
        test_data = pickle.load(f)
    
    with open(data_dir / 'item_metadata.pkl', 'rb') as f:
        item_metadata = pickle.load(f)
    
    num_users = dataset_info['num_users']
    num_items = dataset_info['num_items']
    
    print(f"  Users: {num_users}")
    print(f"  Items: {num_items}")
    print(f"  Test samples: {len(test_data)}")
    
    # Create test data loader
    print("\nCreating test data loader...")
    print(f"  Using eval_seed={eval_seed} for negative sample generation")
    data_module = RecDataModule(
        train_data={},  # Not needed for test
        val_data={},    # Not needed for test
        test_data=test_data,
        item_metadata=item_metadata,
        num_items=num_items,
        tokenizer=None,  # No tokenizer needed for Stage A
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        negative_samples=100,  # For evaluation (SASRec protocol)
        max_seq_length=config['data']['max_sequence_length'],
        max_text_length=None,  # No content loss
        eval_seed=eval_seed  # Ensure same negative samples across runs
    )
    
    test_loader = data_module.test_dataloader()
    
    # Load model
    print(f"\nLoading Stage A model from {args.checkpoint}...")
    model = load_model(args.checkpoint, config, num_users, num_items)
    
    # Evaluate on test data
    print("\n" + "=" * 60)
    print("Evaluating on test data...")
    print("=" * 60)
    
    metrics = evaluate_on_test_data(
        model=model,
        test_loader=test_loader,
        device=args.device,
        max_batches=args.max_batches
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("Test Results:")
    print("=" * 60)
    print_metrics(metrics, prefix="Test ")
    
    # Save results if output file is specified
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Stage A model on test data')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file (optional, will use config from checkpoint if available)')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to Stage A checkpoint directory')
    parser.add_argument('--data_dir', type=str, default='./data/processed',
                       help='Directory with preprocessed data')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of workers for data loading')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--max_batches', type=int, default=None,
                       help='Maximum number of batches to evaluate (None for all)')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Optional path to save evaluation results as JSON')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42, must match dataset seed)')
    
    args = parser.parse_args()
    main(args)

