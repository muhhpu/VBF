import torch
import torch.nn as nn
import torch.nn.functional as F
from variational_encoder import VariationalEncoder
from context_prior import ContextAwarePrior
from dual_decoder import DualDecoder
from meta_learner import MetaLearner


class VariationalFusionLayer(nn.Module):
    """变分贝叶斯融合层"""

    def __init__(self, video_dim=64, audio_dim=64, title_dim=32,
                 latent_dim=32, output_dim=64, num_clusters=8, hidden_dim=128):
        super(VariationalFusionLayer, self).__init__()

        self.input_dim = video_dim + audio_dim + title_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        # 初始化各个组件
        self.encoder = VariationalEncoder(video_dim, audio_dim, title_dim, latent_dim, hidden_dim)
        self.prior_learner = ContextAwarePrior(self.input_dim, latent_dim, num_clusters, hidden_dim)
        self.decoder = DualDecoder(latent_dim, output_dim, hidden_dim)
        self.meta_learner = MetaLearner(self.input_dim, latent_dim, hidden_dim)

        # 融合权重生成网络
        self.fusion_weight_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
            nn.Softmax(dim=1)
        )

        # 最终输出投影
        self.output_projection = nn.Sequential(
            nn.Linear(output_dim + self.input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

        # 损失权重参数 - 固定值而不是可学习参数
        self.kl_weight = 0.001  # 减小KL权重
        self.recon_weight = 1.0

    def compute_kl_divergence(self, mean, logvar, prior_mean, prior_logvar):
        """
        计算KL散度损失 - 修复数值稳定性问题
        """
        # 计算方差
        var = torch.exp(logvar)
        prior_var = torch.exp(prior_logvar)

        # 防止数值不稳定
        var = torch.clamp(var, min=1e-8, max=100)
        prior_var = torch.clamp(prior_var, min=1e-8, max=100)

        # KL散度计算
        kl_div = 0.5 * (
                prior_logvar - logvar +
                var / prior_var +
                (mean - prior_mean).pow(2) / prior_var - 1
        )

        # 对每个样本求和，然后取平均
        kl_loss = torch.mean(torch.sum(kl_div, dim=1))

        # 限制KL损失的范围
        kl_loss = torch.clamp(kl_loss, min=0, max=10)

        return kl_loss

    def adaptive_fusion(self, video_features, audio_features, title_features, fusion_strategy):
        """自适应多模态融合"""
        fusion_weights = self.fusion_weight_net(fusion_strategy)

        weighted_video = video_features * fusion_weights[:, 0:1]
        weighted_audio = audio_features * fusion_weights[:, 1:2]
        weighted_title = title_features * fusion_weights[:, 2:3]

        fused_features = torch.cat([weighted_video, weighted_audio, weighted_title], dim=1)

        return fused_features, fusion_weights

    def forward(self, video_features, audio_features, title_features, use_meta_learning=False):
        """变分贝叶斯融合前向传播"""
        combined_features = torch.cat([video_features, audio_features, title_features], dim=1)

        # 先验学习
        prior_mean, prior_logvar = self.prior_learner(combined_features)

        # 变分编码
        fusion_strategy, mean, logvar = self.encoder(video_features, audio_features, title_features)

        # 自适应融合
        fused_features, fusion_weights = self.adaptive_fusion(
            video_features, audio_features, title_features, fusion_strategy
        )

        # 解码
        decoded_repr, disc_score, quality_score = self.decoder(fusion_strategy)

        # 最终表示
        combined_repr = torch.cat([decoded_repr, fused_features], dim=1)
        final_representation = self.output_projection(combined_repr)

        # 损失计算 - 修复计算方式
        loss_dict = {}

        if self.training:
            # 重构损失：使用余弦相似度而不是MSE
            fused_norm = F.normalize(fused_features, p=2, dim=1)
            combined_norm = F.normalize(combined_features, p=2, dim=1)
            recon_loss = 1 - F.cosine_similarity(fused_norm, combined_norm, dim=1).mean()

            # KL散度损失
            kl_loss = self.compute_kl_divergence(mean, logvar, prior_mean, prior_logvar)

            # 正则化损失
            reg_loss = 0.01 * (torch.norm(fusion_strategy, p=2) + torch.norm(fusion_weights, p=2))

            loss_dict['reconstruction'] = recon_loss
            loss_dict['kl_divergence'] = kl_loss
            loss_dict['regularization'] = reg_loss
            loss_dict['total'] = self.recon_weight * recon_loss + self.kl_weight * kl_loss + reg_loss

        else:
            # 推理时不计算损失
            loss_dict['reconstruction'] = 0.0
            loss_dict['kl_divergence'] = 0.0
            loss_dict['regularization'] = 0.0
            loss_dict['total'] = 0.0

        # 信息字典
        info_dict = {
            'fusion_strategy': fusion_strategy.detach(),
            'fusion_weights': fusion_weights.detach(),
            'quality_score': quality_score.detach(),
            'prior_mean': prior_mean.detach(),
            'prior_logvar': prior_logvar.detach(),
            'posterior_mean': mean.detach(),
            'posterior_logvar': logvar.detach()
        }

        return final_representation, loss_dict, info_dict