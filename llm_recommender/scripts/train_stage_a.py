"""Script to train Stage A (embedding pretraining)."""

import sys
import argparse
import pickle
import json
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import torch
from torch.optim import AdamW
from transformers import AutoTokenizer

from models import StageAModel
from data import RecDataModule
from trainers import StageATrainer
from utils.config import load_config, print_config


def main(args):
    """Main training function for Stage A."""
    # Load config
    config = load_config(args.config)
    
    print("=" * 60)
    print("Stage A: Embedding Pretraining")
    print("=" * 60)
    print("\nConfiguration:")
    print_config(config)
    
    # Load preprocessed data
    data_dir = Path(args.data_dir)
    
    print(f"\nLoading preprocessed data from {data_dir}...")
    
    with open(data_dir / 'train_data.pkl', 'rb') as f:
        train_data = pickle.load(f)
    
    with open(data_dir / 'val_data.pkl', 'rb') as f:
        val_data = pickle.load(f)
    
    with open(data_dir / 'dataset_info.pkl', 'rb') as f:
        dataset_info = pickle.load(f)
    
    with open(data_dir / 'item_metadata.pkl', 'rb') as f:
        item_metadata = pickle.load(f)
    
    num_users = dataset_info['num_users']
    num_items = dataset_info['num_items']
    
    print(f"  Users: {num_users}")
    print(f"  Items: {num_items}")
    print(f"  Train samples: {len(train_data)}")
    print(f"  Val samples: {len(val_data)}")
    
    # Create data module
    print("\nCreating data loaders...")
    data_module = RecDataModule(
        train_data=train_data,
        val_data=val_data,
        test_data=val_data,  # Use val for now
        item_metadata=item_metadata,
        num_items=num_items,
        tokenizer=None,  # No tokenizer needed without content loss
        batch_size=config['stage_a']['batch_size'],
        num_workers=4,
        negative_samples=config['stage_a']['collaborative']['negative_samples'],
        max_seq_length=config['data']['max_sequence_length'],
        max_text_length=None  # No content loss
    )
    
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    
    # Create model
    print("\nInitializing Stage A model...")
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
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=float(config['stage_a']['learning_rate']),
        weight_decay=float(config['stage_a']['weight_decay'])
    )
    
    # Create trainer
    trainer = StageATrainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        device=config['training']['device'],
        loss_weights=config['stage_a']['loss_weights'],
        logging_steps=config['training']['logging_steps'],
        eval_steps=config['training']['eval_steps'],
        save_steps=config['training']['save_steps'],
        output_dir=args.output_dir,
        max_grad_norm=config['stage_a']['max_grad_norm'],
        mixed_precision=config['training']['mixed_precision']
    )
    
    # Save config to checkpoint directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    config_path = output_path / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\nConfiguration saved to {config_path}")
    
    # Load checkpoint if provided
    start_epoch = 0
    if args.resume_from:
        checkpoint_path = Path(args.resume_from)
        if checkpoint_path.exists():
            print(f"\nLoading checkpoint from {checkpoint_path}...")
            trainer.load_checkpoint(str(checkpoint_path))
            # Extract epoch from global_step (approximate)
            # Assuming each epoch has roughly the same number of steps
            steps_per_epoch = len(train_loader)
            start_epoch = trainer.global_step // steps_per_epoch
            print(f"Resuming from epoch ~{start_epoch + 1}, global_step={trainer.global_step}")
        else:
            print(f"WARNING: Checkpoint path {checkpoint_path} does not exist. Starting from scratch.")
    
    # Training loop
    print("\n" + "=" * 60)
    print("Starting Stage A training...")
    print("=" * 60)
    
    for epoch in range(start_epoch, config['stage_a']['epochs']):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{config['stage_a']['epochs']}")
        print(f"{'='*60}")
        
        # Train
        train_metrics = trainer.train_epoch(epoch + 1)
        
        print(f"\nEpoch {epoch + 1} training metrics:")
        print(f"  Total loss: {train_metrics.get('loss', 0):.4f}")
        if 'collab_ce' in train_metrics:
            print(f"  Collaborative CE: {train_metrics['collab_ce']:.4f}")
        if 'bpr' in train_metrics:
            print(f"  BPR loss: {train_metrics['bpr']:.4f}")
        if 'regularization' in train_metrics:
            print(f"  Regularization: {train_metrics['regularization']:.4f}")
        
        # Evaluate
        val_metrics = trainer.evaluate()
        print(f"\nEpoch {epoch + 1} validation metrics:")
        for key, value in val_metrics.items():
            print(f"  {key}: {value:.4f}")

        # Save per-epoch train/validation metrics to JSON in the checkpoint directory
        trainer.log_epoch_metrics(epoch + 1, train_metrics, val_metrics)
    
    print("\n" + "=" * 60)
    print("Stage A training complete!")
    print(f"Best Hit@10: {trainer.best_metric:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Stage A')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--data_dir', type=str, default='./data/processed',
                       help='Directory with preprocessed data')
    parser.add_argument('--output_dir', type=str, default='./checkpoints/stage_a',
                       help='Output directory for checkpoints')
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Path to checkpoint directory to resume from (e.g., ./checkpoints/stage_a/checkpoint-10000)')
    
    args = parser.parse_args()
    main(args)

