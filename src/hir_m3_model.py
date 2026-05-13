import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureEmbedding(nn.Module):
    """
    Projects numerical features into a D-dimensional embedding space.
    Extensible to categorical embeddings if needed.
    """
    def __init__(self, num_features, embed_dim):
        super().__init__()
        # Each feature gets its own linear projection to embed_dim
        self.embeddings = nn.ModuleList([
            nn.Linear(1, embed_dim) for _ in range(num_features)
        ])
        
    def forward(self, x):
        # x: (Batch, NumFeatures)
        out = []
        for i, embed_layer in enumerate(self.embeddings):
            # x[:, i:i+1] -> (Batch, 1) -> (Batch, EmbedDim)
            out.append(embed_layer(x[:, i:i+1]))
        # stack to (Batch, NumFeatures, EmbedDim)
        return torch.stack(out, dim=1)

class HierarchicalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=0.1)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(0.1)
        )
        
    def forward(self, x):
        # x: (Batch, NumFeatures, EmbedDim)
        # 1. MHSA + Residual + Norm
        attn_output, attn_weights = self.mha(x, x, x, average_attn_weights=True)
        x = self.norm1(x + attn_output)
        
        # 2. FeedForward + Residual + Norm
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        return x, attn_weights

class HIRModel(nn.Module):
    def __init__(self, num_features, embed_dim=64, num_heads=8, hidden_dim=128):
        super().__init__()
        self.embedding = FeatureEmbedding(num_features, embed_dim)
        self.attention = HierarchicalAttention(embed_dim, num_heads)
        
        # Prediction Head
        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x):
        # x: (Batch, NumFeatures)
        embeddings = self.embedding(x) # (Batch, N, D)
        
        # MHSA
        feat_repr, attn_weights = self.attention(embeddings) # (Batch, N, D), (Batch, N, N)
        
        # Global Pooling (Average across features)
        pooled = feat_repr.mean(dim=1) # (Batch, D)
        
        # Risk Prediction
        logits = self.head(pooled)
        return logits, attn_weights

def compute_hir_penalty(attn_weights, meso_idxs, macro_idxs):
    """
    Calculates the penalty for Meso-Meso and Macro-Macro interactions.
    attn_weights: (Batch, N, N)
    """
    penalty = 0.0
    
    if len(meso_idxs) > 1:
        # Extract Meso-Meso block
        meso_attn = attn_weights[:, meso_idxs, :][:, :, meso_idxs]
        penalty += meso_attn.mean()
        
    if len(macro_idxs) > 1:
        # Extract Macro-Macro block
        macro_attn = attn_weights[:, macro_idxs, :][:, :, macro_idxs]
        penalty += macro_attn.mean()
        
    return penalty
