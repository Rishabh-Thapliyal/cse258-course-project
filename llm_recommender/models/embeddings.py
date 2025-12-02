"""Embedding modules for users and items with projection to LLM space."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CollaborativeEmbedding(nn.Module):
    """Collaborative embeddings for users and items with projection to LLM hidden space."""
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        llm_hidden_size: int,
        lambda_c: float = 0.01,
        init_method: str = 'normal'
    ):
        """
        Args:
            num_users: Number of users
            num_items: Number of items (note: item_id=0 is reserved for padding)
            embedding_dim: Embedding dimension for collaborative space
            llm_hidden_size: Hidden size of LLM (for projection)
            lambda_c: Regularization parameter for collaborative embeddings
            init_method: Initialization method ('normal' or 'xavier')
        """
        super().__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.llm_hidden_size = llm_hidden_size
        self.lambda_c = lambda_c
        
        # Collaborative embedding tables (separate from LLM vocab)
        # Note: num_items includes padding index 0
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim, padding_idx=0)
        
        # Projection layers to LLM hidden space
        self.user_proj = nn.Linear(embedding_dim, llm_hidden_size, bias=False)
        self.item_proj = nn.Linear(embedding_dim, llm_hidden_size, bias=False)
        
        # Initialize embeddings
        self._initialize_embeddings(init_method)
        
        # Print embedding shapes
        print(f"CollaborativeEmbedding initialized:")
        print(f"  User embeddings: {self.user_embeddings.weight.shape}")
        print(f"  Item embeddings: {self.item_embeddings.weight.shape}")
        print(f"  User projection: {self.user_proj.weight.shape}")
        print(f"  Item projection: {self.item_proj.weight.shape}")
    
    def _initialize_embeddings(self, method: str):
        """Initialize embeddings."""
        if method == 'normal':
            # N(0, lambda_c^-1 * I) as in CLLM4Rec
            std = (1.0 / self.lambda_c) ** 0.5
            nn.init.normal_(self.user_embeddings.weight, mean=0, std=std)
            nn.init.normal_(self.item_embeddings.weight, mean=0, std=std)
        elif method == 'xavier':
            nn.init.xavier_uniform_(self.user_embeddings.weight)
            nn.init.xavier_uniform_(self.item_embeddings.weight)
        else:
            raise ValueError(f"Unknown init method: {method}")
        
        # Initialize projection layers
        nn.init.xavier_uniform_(self.user_proj.weight)
        nn.init.xavier_uniform_(self.item_proj.weight)
    
    def get_user_embeddings(self, user_ids: torch.Tensor) -> tuple:
        """
        Get user embeddings (both raw and projected).
        
        Args:
            user_ids: User IDs (batch_size,)
            
        Returns:
            Tuple of (raw_embeds, projected_embeds)
        """
        raw_embeds = self.user_embeddings(user_ids)  # (B, d_rec)
        proj_embeds = self.user_proj(raw_embeds)      # (B, d_model)
        return raw_embeds, proj_embeds
    
    def get_item_embeddings(self, item_ids: torch.Tensor) -> tuple:
        """
        Get item embeddings (both raw and projected).
        
        Args:
            item_ids: Item IDs (batch_size, seq_len) or (batch_size,)
            
        Returns:
            Tuple of (raw_embeds, projected_embeds)
        """
        raw_embeds = self.item_embeddings(item_ids)  # (B, L, d_rec) or (B, d_rec)
        proj_embeds = self.item_proj(raw_embeds)      # (B, L, d_model) or (B, d_model)
        return raw_embeds, proj_embeds
    
    def forward(
        self,
        user_ids: Optional[torch.Tensor] = None,
        item_ids: Optional[torch.Tensor] = None,
        project: bool = True
    ) -> tuple:
        """
        Forward pass.
        
        Args:
            user_ids: User IDs (batch_size,)
            item_ids: Item IDs (batch_size, seq_len) or (batch_size,)
            project: Whether to return projected embeddings
            
        Returns:
            Tuple of (user_embeds, item_embeds)
        """
        user_embeds = None
        item_embeds = None
        
        if user_ids is not None:
            if project:
                _, user_embeds = self.get_user_embeddings(user_ids)
            else:
                user_embeds, _ = self.get_user_embeddings(user_ids)
        
        if item_ids is not None:
            if project:
                _, item_embeds = self.get_item_embeddings(item_ids)
            else:
                item_embeds, _ = self.get_item_embeddings(item_ids)
        
        return user_embeds, item_embeds
    
    def get_all_item_embeddings(self, project: bool = False) -> torch.Tensor:
        """Get all item embeddings (for scoring)."""
        if project:
            return self.item_proj(self.item_embeddings.weight)
        return self.item_embeddings.weight
    
    def regularization_loss(self) -> torch.Tensor:
        """Compute regularization loss for collaborative embeddings."""
        # L2 regularization on embeddings (not projection weights)
        # Normalize by number of embeddings to avoid scaling with dataset size
        user_reg = (self.user_embeddings.weight ** 2).mean()
        item_reg = (self.item_embeddings.weight ** 2).mean()
        return self.lambda_c * (user_reg + item_reg) / 2


class ContentEmbedding(nn.Module):
    """Content embeddings for users and items with projection to LLM space."""
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        llm_hidden_size: int,
        lambda_t: float = 0.01
    ):
        """
        Args:
            num_users: Number of users
            num_items: Number of items
            embedding_dim: Embedding dimension
            llm_hidden_size: Hidden size of LLM (for projection)
            lambda_t: Regularization parameter encouraging content embeddings
                      to stay close to collaborative embeddings
        
        Note: Content embeddings should be initialized via initialize_from_collaborative()
              method after collaborative embeddings are created.
        """
        super().__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.llm_hidden_size = llm_hidden_size
        self.lambda_t = lambda_t
        
        # Content embedding tables (will be initialized from collaborative)
        # Note: num_items includes padding index 0
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim, padding_idx=0)
        
        # Projection layers to LLM hidden space
        self.user_proj = nn.Linear(embedding_dim, llm_hidden_size, bias=False)
        self.item_proj = nn.Linear(embedding_dim, llm_hidden_size, bias=False)
        
        # Initialize projection layers
        nn.init.xavier_uniform_(self.user_proj.weight)
        nn.init.xavier_uniform_(self.item_proj.weight)
        
        # Print embedding shapes (will be initialized later from collaborative)
        print(f"ContentEmbedding initialized:")
        print(f"  User embeddings: {self.user_embeddings.weight.shape}")
        print(f"  Item embeddings: {self.item_embeddings.weight.shape}")
        print(f"  User projection: {self.user_proj.weight.shape}")
        print(f"  Item projection: {self.item_proj.weight.shape}")
    
    def initialize_from_collaborative(
        self,
        user_collab_embeds: torch.Tensor,
        item_collab_embeds: torch.Tensor
    ):
        """
        Initialize content embeddings from collaborative embeddings.
        
        As per Equation (2): e_cont ~ N(e_coll, lambda_t^-1 * I)
        """
        with torch.no_grad():
            # Initialize from collaborative + small noise
            std = (1.0 / self.lambda_t) ** 0.5
            
            self.user_embeddings.weight.copy_(user_collab_embeds)
            self.user_embeddings.weight.add_(
                torch.randn_like(user_collab_embeds) * std
            )
            
            self.item_embeddings.weight.copy_(item_collab_embeds)
            self.item_embeddings.weight.add_(
                torch.randn_like(item_collab_embeds) * std
            )
    
    def get_user_embeddings(self, user_ids: torch.Tensor) -> tuple:
        """Get user embeddings (both raw and projected)."""
        raw_embeds = self.user_embeddings(user_ids)
        proj_embeds = self.user_proj(raw_embeds)
        return raw_embeds, proj_embeds
    
    def get_item_embeddings(self, item_ids: torch.Tensor) -> tuple:
        """Get item embeddings (both raw and projected)."""
        raw_embeds = self.item_embeddings(item_ids)
        proj_embeds = self.item_proj(raw_embeds)
        return raw_embeds, proj_embeds
    
    def forward(
        self,
        user_ids: Optional[torch.Tensor] = None,
        item_ids: Optional[torch.Tensor] = None,
        project: bool = True
    ) -> tuple:
        """Forward pass."""
        user_embeds = None
        item_embeds = None
        
        if user_ids is not None:
            if project:
                _, user_embeds = self.get_user_embeddings(user_ids)
            else:
                user_embeds, _ = self.get_user_embeddings(user_ids)
        
        if item_ids is not None:
            if project:
                _, item_embeds = self.get_item_embeddings(item_ids)
            else:
                item_embeds, _ = self.get_item_embeddings(item_ids)
        
        return user_embeds, item_embeds
    
    def get_all_item_embeddings(self, project: bool = False) -> torch.Tensor:
        """Get all item embeddings (for scoring)."""
        if project:
            return self.item_proj(self.item_embeddings.weight)
        return self.item_embeddings.weight
    
    def alignment_loss(
        self,
        collab_user_embeds: torch.Tensor,
        collab_item_embeds: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute alignment loss to keep content embeddings close to collaborative ones.
        
        L_align = lambda_t / 2 * (||e_cont_user - e_coll_user||^2 + ||e_cont_item - e_coll_item||^2)
        Normalized by number of embeddings to avoid scaling with dataset size.
        """
        user_diff = self.user_embeddings.weight - collab_user_embeds
        item_diff = self.item_embeddings.weight - collab_item_embeds
        
        # Use mean instead of sum to normalize by number of embeddings
        user_loss = (user_diff ** 2).mean()
        item_loss = (item_diff ** 2).mean()
        
        return self.lambda_t * (user_loss + item_loss) / 2


class ContrastiveLoss(nn.Module):
    """Contrastive loss to align collaborative and content embeddings."""
    
    def __init__(self, temperature: float = 0.07):
        """
        Args:
            temperature: Temperature parameter for InfoNCE loss
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        collab_embeds: torch.Tensor,
        content_embeds: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute InfoNCE contrastive loss (Equations 6 & 7).
        
        Args:
            collab_embeds: Collaborative embeddings (batch_size, dim)
            content_embeds: Content embeddings (batch_size, dim)
            
        Returns:
            Contrastive loss
        """
        # Normalize embeddings
        collab_embeds = F.normalize(collab_embeds, dim=-1)
        content_embeds = F.normalize(content_embeds, dim=-1)
        
        # Compute similarity matrix
        similarity = torch.matmul(collab_embeds, content_embeds.t()) / self.temperature
        
        # Labels: diagonal elements are positive pairs
        labels = torch.arange(collab_embeds.size(0), device=collab_embeds.device)
        
        # InfoNCE loss
        loss = F.cross_entropy(similarity, labels)
        
        return loss


class ItemScoringHead(nn.Module):
    """
    Scoring head for next-item prediction.
    Uses dot-product between LLM output and item embeddings.
    Can optionally fuse collaborative and content embeddings.
    """
    
    def __init__(
        self,
        llm_hidden_size: int,
        item_embedding_module: nn.Module,
        content_embedding_module: Optional[nn.Module] = None,
        fusion_weight: float = 0.5,
        use_bias: bool = False
    ):
        """
        Args:
            llm_hidden_size: Hidden size of LLM
            item_embedding_module: Item embedding module (collaborative)
            content_embedding_module: Optional content embedding module for fusion
            fusion_weight: Weight for collaborative embeddings (1-w for content)
            use_bias: Whether to use bias in scoring
        """
        super().__init__()
        
        self.llm_hidden_size = llm_hidden_size
        self.item_embedding_module = item_embedding_module
        self.content_embedding_module = content_embedding_module
        self.fusion_weight = fusion_weight
        
        # Optional projection of LLM hidden state before scoring
        self.output_proj = nn.Linear(llm_hidden_size, item_embedding_module.embedding_dim, bias=use_bias)
        
        nn.init.xavier_uniform_(self.output_proj.weight)
    
    def forward(
        self,
        llm_output: torch.Tensor,
        candidate_items: Optional[torch.Tensor] = None,
        use_fusion: bool = True
    ) -> torch.Tensor:
        """
        Score items based on LLM output.
        
        Args:
            llm_output: LLM hidden state (batch_size, hidden_size)
            candidate_items: Optional candidate item IDs (batch_size, num_candidates)
                           If None, scores all items
            use_fusion: Whether to fuse collaborative and content embeddings
        
        Returns:
            Item scores (batch_size, num_items) or (batch_size, num_candidates)
        """
        # Project LLM output to item embedding space
        query = self.output_proj(llm_output)  # (B, d_rec)
        
        # Get item embeddings (with optional fusion)
        if candidate_items is not None:
            # Score only candidate items
            item_embeds_collab, _ = self.item_embedding_module.get_item_embeddings(candidate_items)  # (B, K, d_rec)
            
            # Fuse with content embeddings if available
            if use_fusion and self.content_embedding_module is not None:
                item_embeds_content, _ = self.content_embedding_module.get_item_embeddings(candidate_items)
                item_embeds = (self.fusion_weight * item_embeds_collab + 
                             (1 - self.fusion_weight) * item_embeds_content)
            else:
                item_embeds = item_embeds_collab
            
            # Batch dot product
            scores = torch.bmm(item_embeds, query.unsqueeze(-1)).squeeze(-1)  # (B, K)
        else:
            # Score all items
            all_item_embeds_collab = self.item_embedding_module.get_all_item_embeddings(project=False)  # (num_items, d_rec)
            
            # Fuse with content embeddings if available
            if use_fusion and self.content_embedding_module is not None:
                all_item_embeds_content = self.content_embedding_module.get_all_item_embeddings(project=False)
                all_item_embeds = (self.fusion_weight * all_item_embeds_collab + 
                                 (1 - self.fusion_weight) * all_item_embeds_content)
            else:
                all_item_embeds = all_item_embeds_collab
            
            scores = torch.matmul(query, all_item_embeds.t())  # (B, num_items)
        
        return scores
