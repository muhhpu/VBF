import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from Fusion_model import Net
import numpy as np


class GCN(torch.nn.Module):
    """
    基于变分贝叶斯融合的图卷积网络
    整合个性化多模态表示学习与图结构建模
    """

    def __init__(self, device, user_num=100, item_num=993,
                 embedding_dim=64, hidden_dim=128, num_layers=2):
        super().__init__()

        # 基本配置
        self.device = device
        self.USER_NUM = user_num
        self.ITEM_NUM = item_num
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 图卷积层
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.convs.append(SAGEConv(embedding_dim, hidden_dim))
            elif i == num_layers - 1:
                self.convs.append(SAGEConv(hidden_dim, embedding_dim))
            else:
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))

        # 变分贝叶斯融合网络
        self.fusion_net = Net(item_num=item_num, video_dim=64, audio_dim=64, title_dim=32)

        # 用户嵌入（可学习参数）
        self.user_embeddings = nn.Parameter(
            torch.nn.init.xavier_uniform_(torch.Tensor(self.USER_NUM, embedding_dim))
        )

        # 批处理配置（用于处理大规模数据）
        self.batch_size = 128  # 每批处理的物品数量

        # Dropout和BatchNorm
        self.dropout = nn.Dropout(0.2)
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim if i < num_layers - 1 else embedding_dim)
            for i in range(num_layers)
        ])

        # 损失权重参数
        self.register_parameter('graph_loss_weight', nn.Parameter(torch.tensor(1.0)))
        self.register_parameter('fusion_loss_weight', nn.Parameter(torch.tensor(1.0)))

    def process_items_in_batches(self, video, audio, title):
        """分批处理物品 - 修复损失处理"""
        item_representations = []
        total_fusion_loss = 0.0
        fusion_info = {'strategies': [], 'weights': [], 'quality_scores': []}

        num_batches = (self.ITEM_NUM + self.batch_size - 1) // self.batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, self.ITEM_NUM)
            batch_indices = list(range(start_idx, end_idx))

            video_batch = torch.from_numpy(video[batch_indices, :]).float().to(self.device)
            audio_batch = torch.from_numpy(audio[batch_indices, :]).float().to(self.device)
            title_batch = torch.from_numpy(title[batch_indices, :]).float().to(self.device)

            item_ids_batch = torch.tensor(batch_indices, dtype=torch.long, device=self.device)

            if self.training:
                batch_repr, loss_dict, info_dict = self.fusion_net(
                    video_batch, audio_batch, title_batch, item_ids_batch
                )

                # 安全地提取损失值
                batch_loss = loss_dict.get('total', 0.0)
                if isinstance(batch_loss, torch.Tensor):
                    batch_loss = batch_loss.item() if batch_loss.numel() == 1 else torch.mean(batch_loss).item()

                total_fusion_loss += batch_loss

                # 收集融合信息
                fusion_info['strategies'].append(info_dict['fusion_strategy'])
                fusion_info['weights'].append(info_dict['fusion_weights'])
                fusion_info['quality_scores'].append(info_dict['quality_score'])

            else:
                batch_repr = self.fusion_net(video_batch, audio_batch, title_batch, item_ids_batch)

            item_representations.append(batch_repr)

        all_item_representations = torch.cat(item_representations, dim=0)
        avg_fusion_loss = total_fusion_loss / num_batches if self.training else 0.0

        return all_item_representations, avg_fusion_loss, fusion_info

    def forward(self, video, audio, title, edge_index):
        """
        前向传播：结合变分贝叶斯融合和图卷积

        Args:
            video, audio, title: 多模态特征 [item_num, feature_dim]
            edge_index: 图边索引 [2, num_edges]

        Returns:
            final_embeddings: 最终节点嵌入 [user_num + item_num, embedding_dim]
            loss_dict: 损失字典（训练时）
        """
        # 1. 分批处理物品，获取变分融合表示
        item_representations, fusion_loss, fusion_info = self.process_items_in_batches(
            video, audio, title
        )

        # 2. 构建完整的节点嵌入矩阵
        # 用户嵌入 + 物品表示
        all_embeddings = torch.cat([self.user_embeddings, item_representations], dim=0)

        # 3. 多层图卷积
        x = all_embeddings
        layer_outputs = []

        for i, conv in enumerate(self.convs):
            # 图卷积
            x = conv(x, edge_index)

            # 批归一化
            if x.size(0) > 1:  # 确保有足够的样本进行批归一化
                x = self.batch_norms[i](x)

            # 激活函数（除了最后一层）
            if i < len(self.convs) - 1:
                x = F.leaky_relu(x, negative_slope=0.2)
                x = self.dropout(x)

            layer_outputs.append(x)

        # 4. 残差连接（如果维度匹配）
        if all_embeddings.size(1) == x.size(1):
            final_embeddings = all_embeddings + x
        else:
            final_embeddings = x

        # 5. 计算总损失（训练时）
        loss_dict = {}
        if self.training:
            # 图结构损失（可选：添加图正则化项）
            graph_loss = self.compute_graph_regularization(final_embeddings, edge_index)

            # 融合损失
            loss_dict['fusion_loss'] = self.fusion_loss_weight * fusion_loss
            loss_dict['graph_loss'] = self.graph_loss_weight * graph_loss
            loss_dict['total_loss'] = loss_dict['fusion_loss'] + loss_dict['graph_loss']

            # 添加融合信息到损失字典
            loss_dict['fusion_info'] = fusion_info

            return final_embeddings, loss_dict
        else:
            return final_embeddings

    def compute_graph_regularization(self, embeddings, edge_index):
        """
        计算图正则化损失：鼓励相邻节点具有相似表示

        Args:
            embeddings: 节点嵌入 [num_nodes, embedding_dim]
            edge_index: 边索引 [2, num_edges]

        Returns:
            reg_loss: 正则化损失
        """
        # 获取边的源节点和目标节点嵌入
        src_embeddings = embeddings[edge_index[0]]
        dst_embeddings = embeddings[edge_index[1]]

        # 计算相邻节点嵌入的距离
        edge_distances = F.mse_loss(src_embeddings, dst_embeddings, reduction='mean')

        # 图平滑正则化
        smoothness_loss = edge_distances

        # 嵌入范数正则化（防止过拟合）
        norm_loss = torch.mean(torch.norm(embeddings, p=2, dim=1))

        total_reg_loss = smoothness_loss + 0.01 * norm_loss

        return total_reg_loss

    def get_user_embeddings(self, user_ids):
        """获取指定用户的嵌入"""
        return self.user_embeddings[user_ids]

    def get_item_embeddings(self, item_ids, video, audio, title):
        """获取指定物品的嵌入"""
        # 提取对应物品的特征
        video_features = torch.from_numpy(video[item_ids, :]).float().to(self.device)
        audio_features = torch.from_numpy(audio[item_ids, :]).float().to(self.device)
        title_features = torch.from_numpy(title[item_ids, :]).float().to(self.device)

        # 通过融合网络
        with torch.no_grad():
            item_representations = self.fusion_net(
                video_features, audio_features, title_features,
                torch.tensor(item_ids, dtype=torch.long, device=self.device)
            )

        return item_representations

    def analyze_fusion_patterns(self, video, audio, title, num_samples=100):
        """
        分析融合模式：用于理解不同物品的融合策略

        Returns:
            analysis_results: 分析结果字典
        """
        analysis_results = {
            'fusion_strategies': [],
            'fusion_weights': [],
            'modality_importance': {'video': [], 'audio': [], 'title': []},
            'uncertainty_scores': []
        }

        # 随机采样物品进行分析
        sample_indices = np.random.choice(self.ITEM_NUM, size=min(num_samples, self.ITEM_NUM), replace=False)

        with torch.no_grad():
            for idx in sample_indices:
                # 提取单个物品特征
                video_feat = torch.from_numpy(video[idx:idx + 1, :]).float().to(self.device)
                audio_feat = torch.from_numpy(audio[idx:idx + 1, :]).float().to(self.device)
                title_feat = torch.from_numpy(title[idx:idx + 1, :]).float().to(self.device)

                # 获取融合策略和权重
                strategy, weights = self.fusion_net.get_fusion_strategy(video_feat, audio_feat, title_feat)

                # 计算不确定性（通过多次采样）
                strategies, outputs = self.fusion_net.sample_diverse_strategies(
                    video_feat, audio_feat, title_feat, num_samples=10
                )
                uncertainty = torch.std(outputs, dim=0).mean().item()

                # 收集结果
                analysis_results['fusion_strategies'].append(strategy.cpu().numpy())
                analysis_results['fusion_weights'].append(weights.cpu().numpy())
                analysis_results['modality_importance']['video'].append(weights[0, 0].item())
                analysis_results['modality_importance']['audio'].append(weights[0, 1].item())
                analysis_results['modality_importance']['title'].append(weights[0, 2].item())
                analysis_results['uncertainty_scores'].append(uncertainty)

        return analysis_results