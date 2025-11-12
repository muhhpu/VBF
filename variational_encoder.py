import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class VariationalEncoder(nn.Module):
    """
    变分编码器：将多模态特征映射到融合策略的变分参数
    实现神经变分推断机制，学习从观测特征到潜在融合策略的概率映射
    """

    def __init__(self, video_dim=64, audio_dim=64, title_dim=32, latent_dim=32, hidden_dim=128):
        super(VariationalEncoder, self).__init__()

        self.video_dim = video_dim
        self.audio_dim = audio_dim
        self.title_dim = title_dim
        self.latent_dim = latent_dim
        self.input_dim = video_dim + audio_dim + title_dim  # 160

        # 多模态特征预处理网络
        self.feature_processor = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # 变分参数生成网络
        # 生成均值参数
        self.mean_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, latent_dim)
        )

        # 生成对数方差参数
        self.logvar_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, latent_dim)
        )

        # 参数初始化
        self._initialize_weights()

    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, video_features, audio_features, title_features):
        """
        编码多模态特征为变分参数

        Args:
            video_features: 视频特征 [batch_size, video_dim]
            audio_features: 音频特征 [batch_size, audio_dim]
            title_features: 标题特征 [batch_size, title_dim]

        Returns:
            mean: 潜在变量均值 [batch_size, latent_dim]
            logvar: 潜在变量对数方差 [batch_size, latent_dim]
        """
        # 多模态特征拼接
        combined_features = torch.cat([video_features, audio_features, title_features], dim=1)

        # 特征预处理
        processed_features = self.feature_processor(combined_features)

        # 生成变分参数
        mean = self.mean_net(processed_features)
        logvar = self.logvar_net(processed_features)

        # 限制对数方差的范围以提高数值稳定性
        logvar = torch.clamp(logvar, min=-10, max=2)

        return mean, logvar

    def reparameterize(self, mean, logvar):
        """
        重参数化技巧：从变分分布中采样

        Args:
            mean: 均值参数 [batch_size, latent_dim]
            logvar: 对数方差参数 [batch_size, latent_dim]

        Returns:
            z: 采样得到的潜在变量 [batch_size, latent_dim]
        """
        if self.training:
            # 训练时使用重参数化技巧
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mean + eps * std
        else:
            # 推理时使用均值
            z = mean

        return z

    def forward(self, video_features, audio_features, title_features):
        """
        前向传播

        Returns:
            z: 融合策略潜在变量
            mean: 变分均值
            logvar: 变分对数方差
        """
        mean, logvar = self.encode(video_features, audio_features, title_features)
        z = self.reparameterize(mean, logvar)

        return z, mean, logvar