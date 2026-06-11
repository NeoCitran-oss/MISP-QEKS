import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class CrossModalAttention(nn.Module):
    def __init__(self, audio_dim=1280, video_dim=256, embed_dim=128):
        super(CrossModalAttention, self).__init__()
        
        self.W_q = nn.Linear(audio_dim, embed_dim)
        self.W_k = nn.Linear(video_dim, embed_dim)
        self.W_v = nn.Linear(video_dim, embed_dim)

        self.final_linear = nn.Linear(embed_dim, audio_dim)  # 输出维度与音频输入维度相同

    def forward(self, audio_embedding, video_embedding):
        """
        audio_embedding: [1, audio_dim]
        video_embedding: [T, 256], T is variable
        """
        # 计算 Q, K, V
        Q = self.W_q(audio_embedding).unsqueeze(0)  # [1, 1, embed_dim]
        K = self.W_k(video_embedding)  # [T, embed_dim]
        V = self.W_v(video_embedding)  # [T, embed_dim]

        # 计算注意力分数
        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) / (K.size(-1) ** 0.5)  # [1, 1, T]
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)  # [1, 1, T]

        # 应用注意力权重
        attended_video = torch.matmul(attention_probs, V).squeeze(0)  # [embed_dim]

        # 将attended_video投影回原始音频维度
        attended_video_projected = self.final_linear(attended_video)  # [audio_dim]

        # 融合结果
        context_vector = attended_video_projected
        # context_vector = attended_video_projected + audio_embedding  # [384]

        return context_vector

class NoiseReductionMask(nn.Module):
    def __init__(self, embed_dim=1280):
        super(NoiseReductionMask, self).__init__()
        self.conv1d_1 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv1d_2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, context_vector, audio_embedding):
        x = context_vector.permute(1, 0).unsqueeze(0)  # [1, embed_dim, T]
        x = self.conv1d_1(x)
        x = self.relu(x)
        x = self.conv1d_2(x)
        mask = self.sigmoid(x)
        mask = mask.squeeze(0).permute(1, 0)  # [T, embed_dim]
        enhanced_audio = audio_embedding * mask + audio_embedding
        return enhanced_audio

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        assert (
            self.head_dim * num_heads == embed_size
        ), "Embedding size needs to be divisible by num_heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(num_heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.num_heads different pieces
        values = values.reshape(N, value_len, self.num_heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.num_heads, self.head_dim)
        queries = query.reshape(N, query_len, self.num_heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Attention QK^T / sqrt(d_k)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.num_heads * self.head_dim
        )
    
        out = self.fc_out(out)
        return out


class MultiModalAdapter(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super(MultiModalAdapter, self).__init__()
        self.self_attn1 = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.self_attn2 = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output1, _ = self.self_attn1(x, x, x)  # (query, key, value)
        x = x + self.dropout(attn_output1)          # Residual connection
        x = self.norm1(x)                           # Layer Normalization
        attn_output2, _ = self.self_attn2(x, x, x)  # (query, key, value)
        x = x + self.dropout(attn_output2)          # Residual connection
        x = self.norm2(x)                           # Layer Normalization
        return x


class FeedForward(nn.Module):
    def __init__(self, embed_size, ff_hidden_size):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, ff_hidden_size)
        self.fc2 = nn.Linear(ff_hidden_size, embed_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads, ff_hidden_size, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(embed_size, num_heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = FeedForward(embed_size, ff_hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

class Transformer_self(nn.Module):
    def __init__(self, embed_size, num_layers, num_heads, ff_hidden_size, dropout):
        super(Transformer_self, self).__init__()
        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size, num_heads, ff_hidden_size, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, x, x, mask)
        return x

class Transformer_cross(nn.Module):
    def __init__(self, embed_size, num_layers, num_heads, ff_hidden_size, dropout):
        super(Transformer_cross, self).__init__()
        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size, num_heads, ff_hidden_size, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, v, k, q, mask = None):
        for layer in self.layers:
            x = layer(v, k, q, mask)
        return x


class Transformer_encoder(nn.Module):
    def __init__(self, d_model = 128, 
                        nlayers = 2, 
                        nhead = 1, 
                        dim_feedforward = 512,
                        dropout=0.1):
        super(Transformer_encoder, self).__init__()
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

    def forward(self, output, mask = None):#src_key_padding_mask mask
        output = self.transformer_encoder(output, mask)
        return output

class Transformer_encoder_pad(nn.Module):
    def __init__(self, d_model = 100, 
                        nlayers = 2, 
                        nhead = 1, 
                        dim_feedforward = 512,
                        dropout=0.1):
        super(Transformer_encoder_pad, self).__init__()
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

    def forward(self, output, mask = None):#src_key_padding_mask mask
        output = self.transformer_encoder(output, mask)
        return output


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(CrossAttentionLayer, self).__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        
        # 前馈神经网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        # Cross-attention mechanism
        attn_output, _ = self.cross_attn(query, key, value, attn_mask = mask)
        query = query + self.dropout1(attn_output)
        query = self.norm1(query)
        
        # Feedforward neural network
        ff_output = self.linear2(self.dropout(F.relu(self.linear1(query))))
        query = query + self.dropout2(ff_output)
        query = self.norm2(query)
        
        return query

