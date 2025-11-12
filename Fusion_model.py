# Fusion_model.py 的修改版本
import torch
import torch.nn as nn
from variational_fusion import VariationalFusionLayer


class Net(nn.Module):
    """
    基于变分贝叶斯推断的个性化多模态表示学习网络
    """

    def __init__(self, item_num=993, video_dim=64, audio_dim=64, title_dim=32):
        super(Net, self).__init__()

        # 网络配置参数
        self.item_num = item_num
        self.video_dim = video_dim
        self.audio_dim = audio_dim
        self.title_dim = title_dim
        self.latent_dim = 32  # 融合策略潜在空间维度
        self.output_dim = 64  # 最终表示维度

        # 核心变分贝叶斯融合层
        self.variational_fusion = VariationalFusionLayer(
            video_dim=video_dim,
            audio_dim=audio_dim,
            title_dim=title_dim,
            latent_dim=self.latent_dim,
            output_dim=self.output_dim,
            num_clusters=8,
            hidden_dim=128
        )

        # 物品嵌入矩阵（可学习的物品特定表示）
        self.item_embeddings = nn.Parameter(
            torch.nn.init.xavier_uniform_(torch.Tensor(self.item_num, 32))
        )

        # 修改：确保最终输出是64维
        self.final_projection = nn.Sequential(
            nn.Linear(self.output_dim + 32, self.output_dim),  # 96 -> 64
            nn.LayerNorm(self.output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.output_dim, self.output_dim)  # 64 -> 64
        )

        # 自注意力机制
        self.self_attention = nn.MultiheadAttention(
            embed_dim=self.output_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        # 层归一化
        self.layer_norm = nn.LayerNorm(self.output_dim)

        # 训练模式标志
        self.use_meta_learning = False

    def set_meta_learning_mode(self, use_meta=True):
        """设置是否使用元学习模式"""
        self.use_meta_learning = use_meta

    def forward(self, video, audio, title, item_id):
        """
        前向传播：实现个性化多模态表示学习

        Args:
            video: 视频特征 [batch_size, video_dim]
            audio: 音频特征 [batch_size, audio_dim]
            title: 标题特征 [batch_size, title_dim]
            item_id: 物品ID [batch_size]

        Returns:
            final_output: 最终的物品表示 [batch_size, 64]  # 确保是64维
            loss_dict: 训练损失字典（训练时）
            info_dict: 额外信息字典（调试用）
        """
        batch_size = video.size(0)

        # 确保item_id是长整型张量
        if isinstance(item_id, torch.Tensor):
            item_id = item_id.long()
        else:
            item_id = torch.tensor(item_id, dtype=torch.long, device=video.device)

        # 1. 变分贝叶斯多模态融合
        fused_representation, loss_dict, info_dict = self.variational_fusion(
            video, audio, title, use_meta_learning=self.use_meta_learning
        )

        # 2. 获取物品特定嵌入
        item_embeddings = self.item_embeddings[item_id]  # [batch_size, 32]

        # 3. 结合融合表示和物品嵌入
        combined_repr = torch.cat([fused_representation, item_embeddings], dim=1)  # [batch_size, 96]

        # 4. 最终投影到64维
        projected_repr = self.final_projection(combined_repr)  # [batch_size, 64]

        # 5. 自注意力增强（可选）
        if self.training and projected_repr.size(0) > 1:
            # 重塑为序列格式用于自注意力
            seq_repr = projected_repr.unsqueeze(1)  # [batch_size, 1, 64]

            # 应用自注意力
            attended_repr, attention_weights = self.self_attention(
                seq_repr, seq_repr, seq_repr
            )

            # 残差连接和层归一化
            enhanced_repr = self.layer_norm(projected_repr + attended_repr.squeeze(1))

            # 更新信息字典
            info_dict['attention_weights'] = attention_weights
        else:
            enhanced_repr = projected_repr

        # 6. 最终输出 - 确保是64维
        final_output = enhanced_repr  # [batch_size, 64]

        # 训练时返回损失信息，推理时只返回表示
        if self.training:
            return final_output, loss_dict, info_dict
        else:
            return final_output

    def get_fusion_strategy(self, video, audio, title):
        """获取融合策略（用于分析和可视化）"""
        with torch.no_grad():
            strategy, mean, logvar = self.variational_fusion.encoder(video, audio, title)
            weights = self.variational_fusion.fusion_weight_net(strategy)
            return strategy, weights

    def sample_diverse_strategies(self, video, audio, title, num_samples=5):
        """为给定输入采样多种融合策略"""
        strategies = []
        outputs = []

        with torch.no_grad():
            mean, logvar = self.variational_fusion.encoder.encode(video, audio, title)

            for _ in range(num_samples):
                strategy = self.variational_fusion.encoder.reparameterize(mean, logvar)
                strategies.append(strategy)

                fused_features, _ = self.variational_fusion.adaptive_fusion(
                    video, audio, title, strategy
                )
                decoded_repr, _, _ = self.variational_fusion.decoder(strategy)
                combined_repr = torch.cat([decoded_repr, fused_features], dim=1)
                output = self.variational_fusion.output_projection(combined_repr)
                outputs.append(output)

        return torch.stack(strategies), torch.stack(outputs)