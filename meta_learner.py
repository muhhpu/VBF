import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class MetaLearner(nn.Module):
    """
    元学习驱动的快速适应模块
    学习如何快速为新视频生成初始的融合策略，实现few-shot适应
    """

    def __init__(self, input_dim=160, latent_dim=32, hidden_dim=128, num_inner_steps=3):
        super(MetaLearner, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_inner_steps = num_inner_steps

        # 元网络：快速生成初始融合策略
        self.meta_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # 均值和对数方差
        )

        # 适应网络：针对特定视频进行微调
        self.adaptation_net = nn.ModuleDict({
            'layer1': nn.Linear(input_dim + latent_dim, hidden_dim),
            'layer2': nn.Linear(hidden_dim, hidden_dim // 2),
            'layer3': nn.Linear(hidden_dim // 2, latent_dim * 2)
        })

        # 更新策略网络：学习如何更新参数
        self.update_net = nn.Sequential(
            nn.Linear(latent_dim * 4, hidden_dim),  # 当前策略+梯度信息
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # 新的策略参数
        )

        # 学习率生成网络
        self.lr_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def generate_initial_strategy(self, features):
        """
        为新视频生成初始融合策略

        Args:
            features: 视频特征 [batch_size, input_dim]

        Returns:
            initial_mean: 初始策略均值 [batch_size, latent_dim]
            initial_logvar: 初始策略对数方差 [batch_size, latent_dim]
        """
        strategy_params = self.meta_net(features)

        initial_mean = strategy_params[:, :self.latent_dim]
        initial_logvar = strategy_params[:, self.latent_dim:]

        # 限制对数方差范围
        initial_logvar = torch.clamp(initial_logvar, min=-3, max=1)

        return initial_mean, initial_logvar

    def adapt_strategy(self, features, initial_strategy, target_feedback=None):
        """
        针对特定视频适应融合策略

        Args:
            features: 视频特征 [batch_size, input_dim]
            initial_strategy: 初始策略 [batch_size, latent_dim]
            target_feedback: 目标反馈信号（可选）

        Returns:
            adapted_strategy: 适应后的策略
        """
        current_strategy = initial_strategy

        for step in range(self.num_inner_steps):
            # 组合特征和当前策略
            combined_input = torch.cat([features, current_strategy], dim=1)

            # 通过适应网络
            x = F.relu(self.adaptation_net['layer1'](combined_input))
            x = F.relu(self.adaptation_net['layer2'](x))
            strategy_update = self.adaptation_net['layer3'](x)

            # 更新策略
            update_mean = strategy_update[:, :self.latent_dim]
            update_logvar = strategy_update[:, self.latent_dim:]

            # 学习率调制
            lr = self.lr_net(features) * 0.1  # 限制学习率范围

            # 策略更新
            current_strategy = current_strategy + lr * update_mean

            # 如果有目标反馈，进行监督更新
            if target_feedback is not None:
                feedback_loss = F.mse_loss(current_strategy, target_feedback)
                # 计算梯度并更新（简化版本）
                strategy_grad = torch.autograd.grad(
                    feedback_loss, current_strategy,
                    retain_graph=True, create_graph=True
                )[0]
                current_strategy = current_strategy - lr * strategy_grad

        return current_strategy

    def meta_update(self, support_features, support_strategies, query_features, query_strategies):
        """
        元学习更新：基于支持集和查询集更新元网络

        Args:
            support_features: 支持集特征
            support_strategies: 支持集策略
            query_features: 查询集特征
            query_strategies: 查询集策略

        Returns:
            meta_loss: 元学习损失
        """
        # 1. 使用支持集生成初始策略
        support_initial_mean, support_initial_logvar = self.generate_initial_strategy(support_features)
        support_initial_strategy = support_initial_mean  # 简化：使用均值

        # 2. 快速适应
        adapted_strategy = self.adapt_strategy(support_features, support_initial_strategy, support_strategies)

        # 3. 在查询集上评估
        query_initial_mean, query_initial_logvar = self.generate_initial_strategy(query_features)
        query_initial_strategy = query_initial_mean

        query_adapted_strategy = self.adapt_strategy(query_features, query_initial_strategy)

        # 4. 计算元损失
        meta_loss = F.mse_loss(query_adapted_strategy, query_strategies)

        return meta_loss, adapted_strategy

    def forward(self, features, num_adaptation_steps=None):
        """
        前向传播：快速生成适应的融合策略
        """
        # 生成初始策略
        initial_mean, initial_logvar = self.generate_initial_strategy(features)
        initial_strategy = initial_mean  # 推理时使用均值

        # 如果指定了适应步数，进行快速适应
        if num_adaptation_steps is not None:
            adapted_strategy = self.adapt_strategy(features, initial_strategy)
            return adapted_strategy, initial_mean, initial_logvar
        else:
            return initial_strategy, initial_mean, initial_logvar