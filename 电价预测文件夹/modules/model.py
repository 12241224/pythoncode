# 修改 model.py 使用更高效的注意力机制
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config
import numpy as np
class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        # 注册为缓冲区，这样它不会被当作模型参数
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # 添加位置编码到输入
        return x + self.pe[:x.size(0), :]
    
class EfficientAttention(nn.Module):
    """更高效的自注意力机制"""
    def __init__(self, d_model, nhead, dropout=0.1):
        super(EfficientAttention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # 线性投影
        q = self.q_proj(x).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        
        # 使用更高效的点积注意力计算
        # 这里使用缩放点积注意力，但可以替换为更高效的变体
        attn_output = F.scaled_dot_product_attention(
            q, k, v, 
            dropout_p=self.dropout.p if self.training else 0
        )
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(attn_output)

class EfficientTransformerForecaster(nn.Module):
    """更高效的Transformer预测模型"""
    def __init__(self, input_size, output_size):
        super(EfficientTransformerForecaster, self).__init__()
        
        self.input_projection = nn.Linear(input_size, config.D_MODEL)
        self.pos_encoder = PositionalEncoding(config.D_MODEL)
        
        # 使用更高效的注意力层
        self.attention_layers = nn.ModuleList([
            EfficientAttention(config.D_MODEL, config.NHEAD, config.DROPOUT)
            for _ in range(config.NUM_LAYERS)
        ])
        
        self.ff_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.D_MODEL, config.D_MODEL * 4),
                nn.ReLU(),
                nn.Dropout(config.DROPOUT),
                nn.Linear(config.D_MODEL * 4, config.D_MODEL)
            )
            for _ in range(config.NUM_LAYERS)
        ])
        
        self.layer_norms1 = nn.ModuleList([
            nn.LayerNorm(config.D_MODEL) for _ in range(config.NUM_LAYERS)
        ])
        
        self.layer_norms2 = nn.ModuleList([
            nn.LayerNorm(config.D_MODEL) for _ in range(config.NUM_LAYERS)
        ])
        
        self.decoder = nn.Linear(config.D_MODEL, output_size)
        
    def forward(self, src):
        # 输入投影
        src = self.input_projection(src)
        
        # 添加位置编码
        src = self.pos_encoder(src)
        
        # 逐层处理
        for i in range(config.NUM_LAYERS):
            # 自注意力
            residual = src
            src = self.layer_norms1[i](src)
            src = self.attention_layers[i](src)
            src = residual + src
            
            # 前馈网络
            residual = src
            src = self.layer_norms2[i](src)
            src = self.ff_layers[i](src)
            src = residual + src
        
        # 解码
        output = self.decoder(src)
        # 只输出未来FUTURE_STEPS步
        output = output[-config.FUTURE_STEPS:]
        return output