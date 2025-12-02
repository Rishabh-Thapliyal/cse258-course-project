"""Trainer for Stage A: Embedding pretraining."""

import os
import json
import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Optional, List

from models import StageAModel
from utils.metrics import compute_metrics, compute_sampled_metrics, print_metrics


class StageATrainer:
    """Trainer for Stage A embedding pretraining."""
    
    def __init__(
        self,
        model: StageAModel,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: str = 'cuda',
        loss_weights: Dict[str, float] = None,
        logging_steps: int = 100,
        eval_steps: int = 2000,
        save_steps: int = 5000,
        output_dir: str = './checkpoints/stage_a',
        max_grad_norm: float = 1.0,
        mixed_precision: str = 'no'
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.device = device
        self.mixed_precision = mixed_precision
        
        # Default loss weights if not provided
        self.loss_weights = loss_weights or {
            'collaborative': 1.0,
            'cf_bpr': 1.0
        }
        
        self.logging_steps = logging_steps
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.output_dir = output_dir
        self.max_grad_norm = max_grad_norm
        
        # Setup mixed precision training
        self.use_amp = mixed_precision in ['fp16', 'bf16']
        self.amp_dtype = torch.float16 if mixed_precision == 'fp16' else (
            torch.bfloat16 if mixed_precision == 'bf16' else torch.float32
        )
        # Determine device type for autocast
        self.device_type = 'cuda' if 'cuda' in device else 'cpu'
        # GradScaler only needed for fp16 on CUDA
        self.scaler = GradScaler(self.device_type) if mixed_precision == 'fp16' and self.device_type == 'cuda' else None
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Path to JSON file for per-epoch metrics
        self.metrics_path: Path = Path(output_dir) / "metrics.json"
        self.epoch_history: List[Dict[str, float]] = []
        
        # Move model to device
        self.model.to(device)
        
        # Training state
        self.global_step = 0
        self.best_metric = 0.0
        
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        loss_components = {}
        
        # Track individual loss components for averaging
        component_sums = {
            'collab_ce': 0.0,
            'bpr': 0.0,
            'regularization': 0.0
        }
        component_counts = {k: 0 for k in component_sums.keys()}
        
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with autocast(device_type=self.device_type, enabled=self.use_amp, dtype=self.amp_dtype):
                # Compute collaborative loss
                is_autoregressive = batch.get('is_autoregressive', False)
                collab_losses = self.model.compute_collaborative_loss(
                    user_ids=batch['user_ids'],
                    item_ids=batch['item_ids'],
                    attention_mask=batch['attention_mask'],
                    target_items=batch['target_item_ids'],
                    is_autoregressive=is_autoregressive,
                    negative_items=batch.get('negative_items')
                )
                
                # Compute regularization losses
                reg_losses = self.model.compute_regularization_losses()
                
                # Combine losses
                loss = 0.0
                
                # Collaborative losses
                if 'collaborative_ce' in collab_losses:
                    loss += self.loss_weights['collaborative'] * collab_losses['collaborative_ce']
                    loss_components['collab_ce'] = collab_losses['collaborative_ce'].item()
                
                if 'bpr' in collab_losses:
                    loss += self.loss_weights.get('cf_bpr', 0.2) * collab_losses['bpr']
                    loss_components['bpr'] = collab_losses['bpr'].item()
                
                # Regularization (with configurable weight)
                reg_weight = self.loss_weights.get('regularization', 1.0)
                if reg_weight > 0:
                    loss += reg_weight * reg_losses['collab_regularization']
                    loss_components['regularization'] = reg_losses['collab_regularization'].item()
                else:
                    # Still track regularization for monitoring, but don't add to loss
                    loss_components['regularization'] = reg_losses['collab_regularization'].item()
            
            # Backward pass with gradient scaling
            if self.scaler is not None:
                # FP16 training
                self.scaler.scale(loss).backward()
                
                # Gradient clipping (unscale first)
                if self.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                # Optimizer step with scaler
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # FP32 or BF16 training
                loss.backward()
                
                # Gradient clipping
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                # Optimizer step
                self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            
            # Track individual components
            if 'collab_ce' in loss_components:
                component_sums['collab_ce'] += loss_components['collab_ce']
                component_counts['collab_ce'] += 1
            if 'bpr' in loss_components:
                component_sums['bpr'] += loss_components['bpr']
                component_counts['bpr'] += 1
            if 'regularization' in loss_components:
                component_sums['regularization'] += loss_components['regularization']
                component_counts['regularization'] += 1
            
            self.global_step += 1
            
            # Logging
            if self.global_step % self.logging_steps == 0:
                avg_loss = total_loss / (batch_idx + 1)
                
                # Print detailed loss breakdown
                print(f"\n[Step {self.global_step}] Loss breakdown:")
                print(f"  Total loss: {loss.item():.4f}")
                if 'collab_ce' in loss_components:
                    print(f"  Collaborative CE: {loss_components['collab_ce']:.4f} (weight: {self.loss_weights.get('collaborative', 1.0)})")
                if 'bpr' in loss_components:
                    print(f"  BPR loss: {loss_components['bpr']:.4f} (weight: {self.loss_weights.get('cf_bpr', 0.2)})")
                if 'regularization' in loss_components:
                    print(f"  Regularization: {loss_components['regularization']:.4f} (weight: {self.loss_weights.get('regularization', 1.0)})")
                
                progress_bar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'step': self.global_step
                })
                
            
            # Evaluation (limit to 1000 batches for speed)
            # if self.global_step % self.eval_steps == 0:
            #     metrics = self.evaluate(max_batches=1000)
            #     print_metrics(metrics, prefix=f"[Step {self.global_step}] Validation ")
            #     self.model.train()
            
            # Checkpointing
            if self.global_step % self.save_steps == 0:
                self.save_checkpoint(f"checkpoint-{self.global_step}")
        
        # Compute average loss components
        avg_components = {}
        for key in component_sums:
            if component_counts[key] > 0:
                avg_components[key] = component_sums[key] / component_counts[key]
        
        return {
            'loss': total_loss / len(self.train_dataloader),
            **avg_components
        }
    
    @torch.no_grad()
    def evaluate(self, max_batches: Optional[int] = None) -> Dict[str, float]:
        """
        Evaluate the model using SASRec-style sampled evaluation.
        
        For each user:
        1. Compute scores over all items
        2. Build candidate set: [positive] + 100 negatives (101 items total)
        3. Gather scores for these 101 candidates
        4. Compute Hit@K / NDCG@K on this sampled list (target index is always 0)
        """
        self.model.eval()
        all_scores = []
        all_target_indices = []
        
        for i, batch in enumerate(tqdm(self.val_dataloader, desc="Evaluating")):
            if max_batches and i >= max_batches:
                break
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Get positive items and negative items
            pos_items = batch['target_item_ids']  # (B,)
            neg_items = batch['negative_items']    # (B, 100) - should be 100 negatives per user
            
            with autocast(device_type=self.device_type, enabled=self.use_amp, dtype=self.amp_dtype):
                # 1) Get LLM output
                llm_output = self.model.forward_collaborative(
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
                candidate_scores = self.model.collab_scoring_head(
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
        metrics = compute_sampled_metrics(all_scores, all_target_indices, k_list=[1,10])
        print_metrics(metrics, prefix=f"[Step {self.global_step}] Validation ")
        
        # Update best metric (using Hit@10)
        if metrics['hit@10'] > self.best_metric:
            self.best_metric = metrics['hit@10']
            self.save_checkpoint("best_model")
            print(f"  ✓ New best model saved! Hit@10: {metrics['hit@10']:.4f}")
        
        return metrics

    def log_epoch_metrics(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ) -> None:
        """Save train/validation metrics for an epoch to a JSON file in the checkpoint directory."""
        record: Dict[str, float] = {
            "epoch": epoch,
            "global_step": float(self.global_step),
        }
        # Prefix train/val keys to avoid collisions
        for key, value in train_metrics.items():
            record[f"train_{key}"] = float(value)
        for key, value in val_metrics.items():
            record[f"val_{key}"] = float(value)

        self.epoch_history.append(record)
        # Write full history to JSON file
        with self.metrics_path.open("w") as f:
            json.dump(self.epoch_history, f, indent=2)
    
    def save_checkpoint(self, checkpoint_name: str):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(self.output_dir, checkpoint_name)
        Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
        
        # Save model state
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'best_metric': self.best_metric
        }, os.path.join(checkpoint_path, 'pytorch_model.pt'))
        
        # Save embeddings separately
        torch.save({
            'collab_embeddings': self.model.collab_embeddings.state_dict(),
        }, os.path.join(checkpoint_path, 'embeddings.pt'))
        
        # Save model source code for reproducibility
        model_file_path = Path(__file__).parent.parent / 'models' / 'stage_a_model.py'
        if model_file_path.exists():
            shutil.copy2(model_file_path, os.path.join(checkpoint_path, 'stage_a_model.py'))
        
        print(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(
            os.path.join(checkpoint_path, 'pytorch_model.pt'),
            map_location='cpu',
            weights_only=False
        )
        
        # Load model state (includes embeddings since they're part of the model)
        missing_keys, unexpected_keys = self.model.load_state_dict(
            checkpoint['model_state_dict'], 
            strict=False
        )
        
        if missing_keys:
            print(f"  WARNING: Missing keys in checkpoint: {len(missing_keys)} keys")
            if len(missing_keys) <= 10:
                for key in missing_keys:
                    print(f"    - {key}")
        if unexpected_keys:
            print(f"  WARNING: Unexpected keys in checkpoint: {len(unexpected_keys)} keys")
            if len(unexpected_keys) <= 10:
                for key in unexpected_keys:
                    print(f"    - {key}")
        
        # Verify embeddings were loaded
        model_keys = set(self.model.state_dict().keys())
        has_collab = any('collab_embeddings' in k for k in model_keys)
        print(f"  Embeddings loaded: collab={has_collab}")
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_metric = checkpoint['best_metric']
        
        print(f"Checkpoint loaded from {checkpoint_path}")
        print(f"  Resuming from global_step={self.global_step}, best_metric={self.best_metric:.4f}")

