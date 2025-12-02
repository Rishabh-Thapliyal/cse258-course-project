"""Evaluation metrics for recommendation."""

import numpy as np
import torch
from typing import Dict, List


def hit_at_k(predictions: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    """
    Compute Hit@K metric.
    
    Args:
        predictions: Predicted item rankings (batch_size, num_items)
        targets: Ground truth items (batch_size,) - item IDs (1-indexed: 1 to num_items)
        k: Top-k
        
    Returns:
        Hit@K score
    """
    _, top_k_indices = torch.topk(predictions, k=k, dim=-1)  # Indices: [0, 1, 2, ..., num_items-1]
    
    # Embedding indices directly correspond to item IDs:
    # - index 0 = item_id 0 (padding, should be ignored)
    # - index i = item_id i
    # So top_k_indices are already item IDs, we just need to filter padding
    
    hits = 0
    valid_targets = 0
    for i, target in enumerate(targets):
        target_id = target.item()
        # Skip if target is padding (0) or invalid
        if target_id <= 0:
            continue
        valid_targets += 1
        
        # Get top-k indices (these are item IDs)
        top_k = top_k_indices[i].cpu().numpy()
        # Filter out padding (index 0)
        top_k = top_k[top_k > 0]
        
        if target_id in top_k:
            hits += 1
    
    return hits / valid_targets if valid_targets > 0 else 0.0


def ndcg_at_k(predictions: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    """
    Compute NDCG@K metric.
    
    Args:
        predictions: Predicted item scores (batch_size, num_items)
        targets: Ground truth items (batch_size,) - item IDs (1-indexed: 1 to num_items)
        k: Top-k
        
    Returns:
        NDCG@K score
    """
    _, top_k_indices = torch.topk(predictions, k=k, dim=-1)  # Indices: [0, 1, 2, ..., num_items-1]
    
    ndcg_scores = []
    for i, target in enumerate(targets):
        target_item = target.item()
        # Skip if target is padding (0) or invalid
        if target_item <= 0:
            continue
            
        top_k = top_k_indices[i].cpu().numpy()
        # Filter out padding (index 0)
        top_k = top_k[top_k > 0]
        
        if target_item in top_k:
            rank = np.where(top_k == target_item)[0][0] + 1
            dcg = 1.0 / np.log2(rank + 1)
            idcg = 1.0 / np.log2(2)  # Perfect ranking (target at position 1)
            ndcg = dcg / idcg
        else:
            ndcg = 0.0
        
        ndcg_scores.append(ndcg)
    
    return np.mean(ndcg_scores) if len(ndcg_scores) > 0 else 0.0


def mrr(predictions: torch.Tensor, targets: torch.Tensor, mrr_max_k: int = 1000) -> float:
    """
    Compute Mean Reciprocal Rank (MRR).
    
    Args:
        predictions: Predicted item scores (batch_size, num_items)
        targets: Ground truth items (batch_size,) - item IDs (1-indexed: 1 to num_items)
        mrr_max_k: Maximum k for topk (to limit memory usage)
        
    Returns:
        MRR score
    """
    max_k = min(mrr_max_k, predictions.size(1))
    _, ranked_indices = torch.topk(predictions, k=max_k, dim=-1)  # Indices: [0, 1, 2, ..., num_items-1]
    
    rr_scores = []
    for i, target in enumerate(targets):
        target_item = target.item()
        # Skip if target is padding (0) or invalid
        if target_item <= 0:
            continue
            
        ranked = ranked_indices[i].cpu().numpy()
        # Filter out padding (index 0)
        ranked = ranked[ranked > 0]
        
        if target_item in ranked:
            rank = np.where(ranked == target_item)[0][0] + 1
            rr = 1.0 / rank
        else:
            rr = 0.0
        
        rr_scores.append(rr)
    
    return np.mean(rr_scores) if len(rr_scores) > 0 else 0.0


def compute_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    k_values: List[int] = [1, 5, 10, 20],
    mrr_max_k: int = 1000
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.
    
    Args:
        predictions: Predicted item scores (batch_size, num_items)
        targets: Ground truth items (batch_size,)
        k_values: List of k values for Hit@K and NDCG@K
        mrr_max_k: Maximum k for MRR topk (to limit memory usage)
        
    Returns:
        Dictionary of metric scores
    """
    metrics = {}
    
    # Hit@K for each k
    for k in k_values:
        metrics[f'hit@{k}'] = hit_at_k(predictions, targets, k)
    
    # NDCG@K for each k
    for k in k_values:
        metrics[f'ndcg@{k}'] = ndcg_at_k(predictions, targets, k)
    
    # MRR
    metrics['mrr'] = mrr(predictions, targets, mrr_max_k=mrr_max_k)
    
    return metrics


def hr_at_k_sampled(scores: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    """
    Compute Hit@K for sampled candidate evaluation (SASRec-style).
    
    Args:
        scores: (B, C) - scores for C candidates (e.g., 101: 1 positive + 100 negatives)
        targets: (B,) - target index in [0, C-1] (typically 0, meaning positive is first)
        k: Top-k
        
    Returns:
        Hit@K score
    """
    topk_idx = scores.topk(k, dim=-1).indices  # (B, k)
    hits = (topk_idx == targets.unsqueeze(-1)).any(dim=-1).float()  # (B,)
    return hits.mean().item()


def ndcg_at_k_sampled(scores: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    """
    Compute NDCG@K for sampled candidate evaluation (SASRec-style).
    
    Args:
        scores: (B, C) - scores for C candidates (e.g., 101: 1 positive + 100 negatives)
        targets: (B,) - target index in [0, C-1] (typically 0, meaning positive is first)
        k: Top-k
        
    Returns:
        NDCG@K score
    """
    topk_idx = scores.topk(k, dim=-1).indices  # (B, k)
    # Check if target is in top-k
    hits = (topk_idx == targets.unsqueeze(-1))  # (B, k)
    
    # Positions 1..k
    positions = torch.arange(1, k + 1, device=scores.device, dtype=torch.float32)
    # Discount = 1 / log2(1 + pos)
    discounts = 1.0 / torch.log2(positions + 1.0)
    
    # For each row, if there is a hit, its discounted gain; else 0
    gains = (hits.float() * discounts.unsqueeze(0)).max(dim=-1).values  # (B,)
    
    # IDCG is 1.0 (perfect ranking: target at position 1)
    idcg = 1.0 / torch.log2(torch.tensor(2.0, device=scores.device))
    ndcg = gains / idcg
    
    return ndcg.mean().item()


def compute_sampled_metrics(
    scores: torch.Tensor,
    targets: torch.Tensor,
    k_list: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """
    Compute SASRec-style sampled metrics.
    
    Args:
        scores: (B, C) - scores for C candidates (e.g., 101: 1 positive + 100 negatives)
        targets: (B,) - target index in [0, C-1] (typically 0, meaning positive is first)
        k_list: List of k values for Hit@K and NDCG@K
        
    Returns:
        Dictionary of metric scores
    """
    scores = scores.to(torch.float32)
    targets = targets.to(torch.long)
    metrics = {}
    
    for k in k_list:
        metrics[f'hit@{k}'] = hr_at_k_sampled(scores, targets, k)
        metrics[f'ndcg@{k}'] = ndcg_at_k_sampled(scores, targets, k)
    
    return metrics


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    """Print metrics in a formatted way."""
    print(f"\n{prefix}Metrics:")
    print("-" * 50)
    for metric_name, value in sorted(metrics.items()):
        print(f"  {metric_name}: {value:.4f}")
    print("-" * 50)

