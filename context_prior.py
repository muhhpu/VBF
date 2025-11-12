import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
import numpy as np


class ContextAwarePrior(nn.Module):
    """
    上下文感知的先验学习机制
    根据视频的语义类别动态调整先验分布，实现分层贝叶斯建模
    """

    def __init__(self, input_dim=160, latent_dim=32, num_clusters=8, hidden_dim=64):
        super(ContextAwarePrior, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_clusters = num_clusters
        self.hidden_dim = hidden_dim

        # 聚类中心（可学习参数）
        self.cluster_centers = nn.Parameter(torch.randn(num_clusters, input_dim))

        # 每个聚类的先验参数网络
        self.prior_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim * 2)  # 均值和对数方差
            ) for _ in range(num_clusters)
        ])

        # 聚类分配网络
        self.assignment_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_clusters)
        )

        # 温度参数用于软分配
        self.temperature = nn.Parameter(torch.tensor(1.0))

        self._initialize_weights()

    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # 随机初始化聚类中心
        nn.init.normal_(self.cluster_centers, std=0.1)

    def get_cluster_assignment(self, features):
        """
        计算特征到各聚类的软分配权重

        Args:
            features: 输入特征 [batch_size, input_dim]

        Returns:
            weights: 软分配权重 [batch_size, num_clusters]
        """
        # 计算到各聚类中心的距离
        distances = torch.cdist(features, self.cluster_centers)  # [batch_size, num_clusters]

        # 使用注意力机制计算分配权重
        assignment_logits = self.assignment_net(features)  # [batch_size, num_clusters]

        # 结合距离和学习的分配逻辑
        combined_logits = assignment_logits - distances * self.temperature

        # 软分配权重
        weights = F.softmax(combined_logits, dim=1)

        return weights

    def get_prior_params(self, features):
        """
        根据输入特征获取上下文感知的先验参数

        Args:
            features: 输入特征 [batch_size, input_dim]

        Returns:
            prior_mean: 先验均值 [batch_size, latent_dim]
            prior_logvar: 先验对数方差 [batch_size, latent_dim]
        """
        batch_size = features.size(0)

        # 获取聚类分配权重
        weights = self.get_cluster_assignment(features)  # [batch_size, num_clusters]

        # 计算每个聚类的先验参数
        all_prior_params = []
        for i, prior_net in enumerate(self.prior_networks):
            cluster_center = self.cluster_centers[i].unsqueeze(0).expand(batch_size, -1)
            prior_params = prior_net(cluster_center)  # [batch_size, latent_dim * 2]
            all_prior_params.append(prior_params)

        all_prior_params = torch.stack(all_prior_params, dim=1)  # [batch_size, num_clusters, latent_dim * 2]

        # 加权组合获得最终先验参数
        weighted_params = torch.sum(weights.unsqueeze(-1) * all_prior_params, dim=1)  # [batch_size, latent_dim * 2]

        # 分离均值和对数方差
        prior_mean = weighted_params[:, :self.latent_dim]
        prior_logvar = weighted_params[:, self.latent_dim:]

        # 限制对数方差范围
        prior_logvar = torch.clamp(prior_logvar, min=-5, max=2)

        return prior_mean, prior_logvar

    def update_clusters(self, features_list):
        """
        使用K-means更新聚类中心（可选的预训练步骤）

        Args:
            features_list: 特征列表用于聚类
        """
        if len(features_list) < self.num_clusters:
            return

        features_array = torch.cat(features_list, dim=0).detach().cpu().numpy()

        # 执行K-means聚类
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=42)
        kmeans.fit(features_array)

        # 更新聚类中心
        new_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
        self.cluster_centers.data = new_centers.to(self.cluster_centers.device)

    def forward(self, features):
        """前向传播"""
        return self.get_prior_params(features)