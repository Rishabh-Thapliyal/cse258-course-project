"""
Stage A Model: Pretrain user/item embeddings with collaborative loss and optional BPR loss.
LLM backbone is frozen in this stage.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from typing import Dict, Optional

from .embeddings import (
    CollaborativeEmbedding,
    ItemScoringHead
)


class StageAModel(nn.Module):
    """
    Stage A: Pretrain user/item token embeddings.
    
    Trains:
    1. Collaborative loss (interaction-only view) - Equation (3)
    2. Optional BPR loss - Equation (4)
    """
    
    def __init__(
        self,
        base_llm_name: str,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        lambda_c: float = 0.01,
        freeze_llm: bool = True,
        use_bpr_loss: bool = True,
        random_init_llm: bool = False
    ):
        """
        Args:
            base_llm_name: Pretrained LLM name from HuggingFace
                          Examples:
                          - GPT-2: 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'
                          - Qwen: 'Qwen/Qwen2-0.5B' (500M), 'Qwen/Qwen2-1.5B' (~1B), 'Qwen/Qwen2-7B'
            num_users: Number of users
            num_items: Number of items
            embedding_dim: Dimension of user/item embeddings (e.g., 64)
                          These are projected to llm_hidden_size (e.g., 768) internally
            lambda_c: Collaborative regularization
            freeze_llm: Whether to freeze LLM parameters
            use_bpr_loss: Whether to use BPR loss
            random_init_llm: If True, initialize LLM weights randomly using config instead of pretrained weights
        """
        super().__init__()
        
        # Load LLM (either pretrained weights or random init from config)
        if random_init_llm:
            llm_config = AutoConfig.from_pretrained(base_llm_name)
            self.llm = AutoModel.from_config(llm_config)
        else:
            self.llm = AutoModel.from_pretrained(base_llm_name)
        self.llm_hidden_size = self.llm.config.hidden_size
        self.vocab_size = self.llm.config.vocab_size
        
        # Freeze LLM in Stage A
        if freeze_llm:
            for param in self.llm.parameters():
                param.requires_grad = False
        
        # Embedding modules
        self.collab_embeddings = CollaborativeEmbedding(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=embedding_dim,
            llm_hidden_size=self.llm_hidden_size,
            lambda_c=lambda_c,
            init_method='xavier'
        )
        
        # Scoring head for collaborative embeddings only (no fusion)
        self.collab_scoring_head = ItemScoringHead(
            self.llm_hidden_size,
            self.collab_embeddings,
            content_embedding_module=None,  # No content embeddings
            fusion_weight=1.0  # Not used when content_embedding_module is None
        )
        
        self.use_bpr_loss = use_bpr_loss
        
        print(f"Stage A Model initialized:")
        print(f"  LLM: {base_llm_name}")
        print(f"  LLM hidden size: {self.llm_hidden_size}")
        print(f"  Vocab size: {self.vocab_size}")
        print(f"  User/Item embedding dim: {embedding_dim} (projected to {self.llm_hidden_size})")
        print(f"  Freeze LLM: {freeze_llm}")
        print(f"  Use BPR loss: {use_bpr_loss}")
        print(f"  Random init LLM: {random_init_llm}")
    
    def forward_collaborative(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for collaborative view.
        
        Constructs sequence: [user_emb, item_emb_1, ..., item_emb_T]
        Feeds through LLM to predict next item.
        
        Args:
            user_ids: (batch_size,)
            item_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            
        Returns:
            LLM output for scoring (batch_size, hidden_size)
        """
        batch_size = user_ids.size(0)
        
        # Get collaborative embeddings (projected to LLM space)
        user_embeds, item_embeds = self.collab_embeddings(
            user_ids=user_ids,
            item_ids=item_ids,
            project=True
        )  # user_embeds: (B, d_model), item_embeds: (B, L, d_model)
        
        # Add user embedding as first token
        user_embeds = user_embeds.unsqueeze(1)  # (B, 1, d_model)
        
        # Concatenate: [user, items]
        inputs_embeds = torch.cat([user_embeds, item_embeds], dim=1)  # (B, L+1, d_model)
        
        # Create attention mask with user token
        user_mask = torch.ones(batch_size, 1, device=attention_mask.device, dtype=attention_mask.dtype)
        full_attention_mask = torch.cat([user_mask, attention_mask], dim=1)  # (B, L+1)
        
        # Pass through LLM
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            output_hidden_states=True
        )
        
        # Get last hidden state
        last_hidden = outputs.last_hidden_state  # (B, L+1, d_model)
        
        # Extract the representation for next-item prediction (last position)
        # Use the last valid position based on attention mask
        seq_lengths = full_attention_mask.sum(dim=1) - 1  # -1 because we want last valid position
        batch_indices = torch.arange(batch_size, device=last_hidden.device)
        output_hidden = last_hidden[batch_indices, seq_lengths]  # (B, d_model)
        
        return output_hidden
    
    def compute_collaborative_loss(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_items: torch.Tensor,
        is_autoregressive: bool = False,
        negative_items: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute collaborative loss (Equation 3) and optional BPR loss (Equation 4).
        
        Supports two modes:
        1. Autoregressive: Predict each next item in sequence (training)
        2. Single-target: Predict one next item (validation/test)
        
        BPR loss is supported in both modes when negative_items are provided.
        
        Args:
            user_ids: (batch_size,)
            item_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            target_items: (batch_size,) for single-target or (batch_size, seq_len) for autoregressive
            is_autoregressive: Whether to use autoregressive training
            negative_items: (batch_size, num_neg) - negative items for BPR loss (both modes)
            
        Returns:
            Dictionary with losses
        """
        batch_size = user_ids.size(0)
        
        # Get collaborative embeddings (projected to LLM space)
        user_embeds, item_embeds = self.collab_embeddings(
            user_ids=user_ids,
            item_ids=item_ids,
            project=True
        )  # user_embeds: (B, d_model), item_embeds: (B, L, d_model)
        
        # Add user embedding as first token
        user_embeds = user_embeds.unsqueeze(1)  # (B, 1, d_model)
        
        # Concatenate: [user, items]
        inputs_embeds = torch.cat([user_embeds, item_embeds], dim=1)  # (B, L+1, d_model)
        
        # Create attention mask with user token
        user_mask = torch.ones(batch_size, 1, device=attention_mask.device, dtype=attention_mask.dtype)
        full_attention_mask = torch.cat([user_mask, attention_mask], dim=1)  # (B, L+1)
        
        # Pass through LLM
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            output_hidden_states=True
        )
        
        last_hidden = outputs.last_hidden_state  # (B, L+1, d_model)
        
        if is_autoregressive:
            # Autoregressive mode: Predict at each position
            # Hidden states: [h_user, h_item1, h_item2, ..., h_itemT]
            # Predictions: predict item2 from h_item1, item3 from h_item2, etc.
            # We use positions 1 to T (skip user position 0)
            
            # Extract hidden states for all item positions (skip user position)
            item_hidden = last_hidden[:, 1:, :]  # (B, L, d_model)
            
            # Project to item embedding space
            h_query = self.collab_scoring_head.output_proj(item_hidden)  # (B, L, d_rec)
            
            # Get item embeddings for scoring (collaborative only)
            all_item_embeds = self.collab_embeddings.get_all_item_embeddings(project=False)  # (num_items, d_rec)
            
            # Compute scores for all positions: (B, L, num_items)
            scores = torch.matmul(h_query, all_item_embeds.t())  # (B, L, num_items)
            
            # Reshape for loss computation
            scores_flat = scores.reshape(-1, scores.size(-1))  # (B*L, num_items)
            targets_flat = target_items.reshape(-1)  # (B*L,)
            
            # Create valid mask: both attention_mask > 0 AND target_items > 0
            # This excludes positions where target=0 (no next item) and padding
            valid_mask = (attention_mask > 0) & (target_items > 0)  # (B, L)
            valid_mask_flat = valid_mask.reshape(-1)  # (B*L,)
            num_valid = valid_mask_flat.sum()
            
            # CrossEntropy with ignore_index for padded positions
            ce_loss = F.cross_entropy(
                scores_flat,
                targets_flat,
                ignore_index=0,  # Ignore padding (item_id=0 is reserved for padding)
                reduction='sum'
            )
            
            # Normalize by number of valid positions (where target > 0)
            ce_loss = ce_loss / num_valid.clamp(min=1)
            
            losses = {'collaborative_ce': ce_loss}
            
            # Optional BPR loss in autoregressive mode (SASRec-style)
            if self.use_bpr_loss and negative_items is not None:
                # 1. Context-item BPR: Use LLM hidden states (projected to embedding space) as context representation
                # This matches SASRec: h = model(seq), then score(h, pos, neg)
                h_context = self.collab_scoring_head.output_proj(item_hidden)  # (B, L, d_rec)
                
                # Get positive and negative item embeddings
                _, target_embeds_raw = self.collab_embeddings(item_ids=target_items, project=False)  # (B, L, d_rec)
                _, neg_embeds_raw = self.collab_embeddings(item_ids=negative_items, project=False)  # (B, num_neg, d_rec)
                
                # Compute positive scores: score(h_context, target) at each position
                pos_scores_context = torch.sum(h_context * target_embeds_raw, dim=-1)  # (B, L)
                
                # Compute negative scores: score(h_context, neg) for each position
                # h_context: (B, L, d_rec), neg_embeds_raw: (B, num_neg, d_rec)
                # Use batch matrix multiplication: (B, L, d_rec) @ (B, d_rec, num_neg) -> (B, L, num_neg)
                neg_scores_context = torch.bmm(h_context, neg_embeds_raw.transpose(1, 2))  # (B, L, num_neg)
                
                # Compute context-item BPR loss: -log(sigmoid(pos - neg) + eps) for each (pos, neg) pair
                pos_scores_expanded = pos_scores_context.unsqueeze(-1)  # (B, L, 1)
                bpr_logits_context = pos_scores_expanded - neg_scores_context  # (B, L, num_neg)
                bpr_loss_context = -torch.log(torch.sigmoid(bpr_logits_context) + 1e-8).sum(dim=-1)  # (B, L) - sum over negatives
                
                # Mask out invalid positions (where target=0) and normalize
                bpr_loss_context = (bpr_loss_context * valid_mask.float()).sum() / num_valid.clamp(min=1)
                losses['bpr_context'] = bpr_loss_context
                
                # 2. User-item BPR: Direct user-item interaction
                # Get user embeddings in raw space (not projected)
                user_embeds_raw, _ = self.collab_embeddings(user_ids=user_ids, project=False)  # (B, d_rec)
                
                # Expand user embeddings to match sequence length: (B, L, d_rec)
                # Each position uses the same user embedding
                user_embeds_expanded = user_embeds_raw.unsqueeze(1).expand(-1, target_embeds_raw.size(1), -1)  # (B, L, d_rec)
                
                # Compute positive scores: score(user, target) at each position
                pos_scores_user = torch.sum(user_embeds_expanded * target_embeds_raw, dim=-1)  # (B, L)
                
                # Compute negative scores: score(user, neg) for each position
                # user_embeds_expanded: (B, L, d_rec), neg_embeds_raw: (B, num_neg, d_rec)
                # Use batch matrix multiplication: (B, L, d_rec) @ (B, d_rec, num_neg) -> (B, L, num_neg)
                neg_scores_user = torch.bmm(user_embeds_expanded, neg_embeds_raw.transpose(1, 2))  # (B, L, num_neg)
                
                # Compute user-item BPR loss: -log(sigmoid(pos - neg) + eps) for each (pos, neg) pair
                pos_scores_user_expanded = pos_scores_user.unsqueeze(-1)  # (B, L, 1)
                bpr_logits_user = pos_scores_user_expanded - neg_scores_user  # (B, L, num_neg)
                bpr_loss_user = -torch.log(torch.sigmoid(bpr_logits_user) + 1e-8).sum(dim=-1)  # (B, L) - sum over negatives
                
                # Mask out invalid positions (where target=0) and normalize
                bpr_loss_user = (bpr_loss_user * valid_mask.float()).sum() / num_valid.clamp(min=1)
                losses['bpr_user'] = bpr_loss_user
                
                # Combined BPR loss (for backward compatibility, can be weighted separately)
                losses['bpr'] = bpr_loss_context + bpr_loss_user
            
        else:
            # Single-target mode: Predict one next item
            # Use last valid position
            seq_lengths = full_attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(batch_size, device=last_hidden.device)
            llm_output = last_hidden[batch_indices, seq_lengths]  # (B, d_model)
            
            # Score all items
            scores = self.collab_scoring_head(llm_output)  # (B, num_items)
            
            # Cross-entropy loss
            ce_loss = F.cross_entropy(scores, target_items)
            
            losses = {'collaborative_ce': ce_loss}
            
            # Optional BPR loss (SASRec-style: using LLM hidden state as context)
            if self.use_bpr_loss and negative_items is not None:
                # 1. Context-item BPR: Use LLM output (projected to embedding space) as context representation
                # This matches SASRec: h = model(seq), then score(h, pos, neg)
                h_context = self.collab_scoring_head.output_proj(llm_output)  # (B, d_rec)
                
                # Get positive and negative item embeddings
                _, target_embeds_raw = self.collab_embeddings(item_ids=target_items, project=False)  # (B, d_rec)
                _, neg_embeds_raw = self.collab_embeddings(item_ids=negative_items, project=False)  # (B, num_neg, d_rec)
                
                # Compute scores: score(h_context, pos/neg)
                pos_scores_context = torch.sum(h_context * target_embeds_raw, dim=-1)  # (B,)
                neg_scores_context = torch.sum(h_context.unsqueeze(1) * neg_embeds_raw, dim=-1)  # (B, num_neg)
                
                # Compute context-item BPR loss: -log(sigmoid(pos - neg) + eps)
                bpr_loss_context = -torch.log(torch.sigmoid(pos_scores_context.unsqueeze(1) - neg_scores_context) + 1e-8).mean()
                losses['bpr_context'] = bpr_loss_context
                
                # 2. User-item BPR: Direct user-item interaction
                # Get user embeddings in raw space (not projected)
                user_embeds_raw, _ = self.collab_embeddings(user_ids=user_ids, project=False)  # (B, d_rec)
                
                # Compute scores: score(user, pos/neg)
                pos_scores_user = torch.sum(user_embeds_raw * target_embeds_raw, dim=-1)  # (B,)
                neg_scores_user = torch.sum(user_embeds_raw.unsqueeze(1) * neg_embeds_raw, dim=-1)  # (B, num_neg)
                
                # Compute user-item BPR loss: -log(sigmoid(pos - neg) + eps)
                bpr_loss_user = -torch.log(torch.sigmoid(pos_scores_user.unsqueeze(1) - neg_scores_user) + 1e-8).mean()
                losses['bpr_user'] = bpr_loss_user
                
                # Combined BPR loss (for backward compatibility, can be weighted separately)
                losses['bpr'] = bpr_loss_context + bpr_loss_user
        
        return losses
    
    def compute_regularization_losses(self) -> Dict[str, torch.Tensor]:
        """Compute regularization losses."""
        collab_reg = self.collab_embeddings.regularization_loss()
        
        return {
            'collab_regularization': collab_reg
        }

