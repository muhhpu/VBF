import torch
import torch.nn as nn
import torch.nn.functional as F


class DualDecoder(nn.Module):
    """
    对偶网络解码器：根据融合策略重构视频表示
    包含解码网络和判别网络，采用对抗学习思想训练
    """

    def __init__(self, latent_dim=32, output_dim=64, hidden_dim=128):
        super(DualDecoder, self).__init__()

        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        # 主解码网络：从融合策略生成视频表示
        self.decoder_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )

        # 判别网络：区分真实和生成的融合策略
        self.discriminator = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 4, 1)
        )

        # 重构质量评估网络
        self.quality_net = nn.Sequential(
            nn.Linear(output_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if 'discriminator' in str(m):
                    # 判别器使用正态分布初始化
                    nn.init.normal_(m.weight, 0.0, 0.02)
                else:
                    # 其他网络使用Xavier初始化
                    nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def decode(self, fusion_strategy):
        """
        从融合策略解码生成视频表示

        Args:
            fusion_strategy: 融合策略潜在变量 [batch_size, latent_dim]

        Returns:
            decoded_representation: 解码得到的视频表示 [batch_size, output_dim]
        """
        return self.decoder_net(fusion_strategy)

    def discriminate(self, fusion_strategy):
        """
        判别融合策略的真实性

        Args:
            fusion_strategy: 融合策略 [batch_size, latent_dim]

        Returns:
            score: 真实性分数 [batch_size, 1]
        """
        return self.discriminator(fusion_strategy)

    def evaluate_quality(self, representation, fusion_strategy):
        """
        评估重构质量

        Args:
            representation: 视频表示 [batch_size, output_dim]
            fusion_strategy: 融合策略 [batch_size, latent_dim]

        Returns:
            quality_score: 质量分数 [batch_size, 1]
        """
        combined = torch.cat([representation, fusion_strategy], dim=1)
        return self.quality_net(combined)

    def forward(self, fusion_strategy):
        """
        前向传播

        Returns:
            decoded_repr: 解码表示
            disc_score: 判别分数
            quality_score: 质量分数
        """
        # 解码生成表示
        decoded_repr = self.decode(fusion_strategy)

        # 判别分数
        disc_score = self.discriminate(fusion_strategy)

        # 质量评估
        quality_score = self.evaluate_quality(decoded_repr, fusion_strategy)

        return decoded_repr, disc_score, quality_score