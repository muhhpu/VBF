import os
import pickle
import torch
from torch import nn
import math
import numpy as np
import torch.utils.data as D
import torch.nn.functional as F
from Fusion_model import Net
from GCN_model import GCN
import scipy.sparse as sparse
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

# 设置环境变量优化内存使用
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


class VariationalBayesianTrainer:
    """
    变分贝叶斯个性化多模态推荐系统训练器
    """

    def __init__(self, config, dataset_name='movielens'):
        self.config = config
        self.dataset_name = dataset_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 加载数据
        self.load_data()

        # 初始化模型
        self.initialize_model()

        # 初始化优化器
        self.initialize_optimizer()

        # 训练统计
        self.train_stats = {
            'total_losses': [],
            'fusion_losses': [],
            'graph_losses': [],
            'kl_losses': [],
            'reconstruction_losses': []
        }

    def load_data(self):
        """加载并验证数据"""
        print(f"Loading data for dataset: {self.dataset_name}")

        # 加载参数文件
        para_file = f'./pro_data/{self.dataset_name}_load.para'
        if not os.path.exists(para_file):
            raise FileNotFoundError(f"Parameter file {para_file} not found!")

        with open(para_file, 'rb') as f_para:
            para_load = pickle.load(f_para)

        self.user_num = para_load['user_num']
        self.item_num = para_load['item_num']
        self.train_ui = para_load['train_ui']

        # 获取数据集的模态信息
        self.has_v = para_load.get('has_v', True)
        self.has_a = para_load.get('has_a', True)
        self.has_t = para_load.get('has_t', True)

        print(f'Total number of users: {self.user_num}')
        print(f'Total number of items: {self.item_num}')
        print(f'Modalities - Video: {self.has_v}, Audio: {self.has_a}, Text: {self.has_t}')

        # 加载PCA处理后的特征数据
        self.load_features()

        # 加载训练三元组
        self.load_training_triplets()

        # 构建图边索引
        self.build_edge_index()

    def load_features(self):
        """加载特征数据"""
        print("Loading features...")

        # 设置特征维度
        if self.dataset_name == 'movielens':
            v_dim, a_dim, t_dim = 64, 64, 32
        elif self.dataset_name == 'tiktok':
            v_dim, a_dim, t_dim = 64, 64, 32
        elif self.dataset_name == 'Kwai':
            v_dim, a_dim, t_dim = 64, 64, 32
        else:
            v_dim, a_dim, t_dim = 64, 64, 32

        # 初始化特征
        self.video_features = None
        self.audio_features = None
        self.title_features = None

        # 加载视频特征
        if self.has_v:
            v_path = f'./pro_feature/{self.dataset_name}_v_{v_dim}.npy'
            if os.path.exists(v_path):
                self.video_features = np.load(v_path)
                print(f"  Video: {self.video_features.shape}")
            else:
                print(f"  Video feature file not found: {v_path}")
                self.has_v = False

        # 加载音频特征
        if self.has_a:
            a_path = f'./pro_feature/{self.dataset_name}_a_{a_dim}.npy'
            if os.path.exists(a_path):
                self.audio_features = np.load(a_path)
                print(f"  Audio: {self.audio_features.shape}")
            else:
                print(f"  Audio feature file not found: {a_path}")
                self.has_a = False

        # 加载文本特征
        if self.has_t:
            t_path = f'./pro_feature/{self.dataset_name}_t_{t_dim}.npy'
            if os.path.exists(t_path):
                self.title_features = np.load(t_path)
                print(f"  Title: {self.title_features.shape}")
            else:
                print(f"  Title feature file not found: {t_path}, generating random features...")
                # 生成随机文本特征作为备选
                self.title_features = np.random.randn(self.item_num, t_dim).astype(np.float32)
                self.has_t = False  # 标记为没有真实的文本特征

        # 如果没有任何特征，生成随机特征
        if self.video_features is None:
            print("  Generating random video features...")
            self.video_features = np.random.randn(self.item_num, v_dim).astype(np.float32)

        if self.audio_features is None:
            print("  Generating random audio features...")
            self.audio_features = np.random.randn(self.item_num, a_dim).astype(np.float32)

        if self.title_features is None:
            print("  Generating random title features...")
            self.title_features = np.random.randn(self.item_num, t_dim).astype(np.float32)

        # 验证数据一致性
        assert self.video_features.shape[0] == self.item_num
        assert self.audio_features.shape[0] == self.item_num
        assert self.title_features.shape[0] == self.item_num

    def load_training_triplets(self):
        """加载训练三元组数据"""
        print("Loading training triplets...")

        train_i = torch.empty(0).long()
        train_j = torch.empty(0).long()
        train_m = torch.empty(0).long()

        data_blocks = self.config.get('data_blocks', 1)

        for b_i in range(data_blocks):
            try:
                triple_file = f'./pro_triple/{self.dataset_name}_triple_{b_i}.para'
                if os.path.exists(triple_file):
                    triple_para = pickle.load(open(triple_file, 'rb'))
                    train_i = torch.cat((train_i, torch.tensor(triple_para['train_i'])))
                    train_j = torch.cat((train_j, torch.tensor(triple_para['train_j'])))
                    train_m = torch.cat((train_m, torch.tensor(triple_para['train_m'])))
                else:
                    print(f"Warning: Triple file {triple_file} not found, generating default triplets")
                    train_i, train_j, train_m = self.generate_default_triplets()
                    break
            except Exception as e:
                print(f"Error loading triple file {b_i}: {e}")
                train_i, train_j, train_m = self.generate_default_triplets()
                break

        # 如果没有加载到任何三元组，生成默认的
        if len(train_i) == 0:
            train_i, train_j, train_m = self.generate_default_triplets()

        # 创建数据加载器
        train_dataset = D.TensorDataset(train_i, train_j, train_m)
        self.train_loader = D.DataLoader(
            dataset=train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True
        )

        print(f"Training triplets loaded: {len(train_dataset)}")

    def generate_default_triplets(self):
        """生成默认的训练三元组（如果文件不存在）"""
        print("Generating default training triplets...")

        triplets = []
        for user_id in range(self.user_num):
            # 为每个用户生成一些随机的正负样本对
            num_interactions = min(10, self.item_num // 2)
            pos_items = np.random.choice(self.item_num, num_interactions, replace=False)

            for pos_item in pos_items:
                # 负采样
                neg_item = np.random.choice(self.item_num)
                while neg_item == pos_item:
                    neg_item = np.random.choice(self.item_num)

                triplets.append([user_id, pos_item, neg_item])

        triplets = np.array(triplets)
        return (torch.tensor(triplets[:, 0]),
                torch.tensor(triplets[:, 1]),
                torch.tensor(triplets[:, 2]))

    def build_edge_index(self):
        """构建图的边索引"""
        print("Building graph edge index...")

        # 构建双向边
        edges = np.concatenate((self.train_ui, self.train_ui[:, [1, 0]]), axis=0)

        # 调整物品索引（物品节点在用户节点之后）
        edges[:, 1] += self.user_num

        self.edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(self.device)

        print(f"Graph edge index shape: {self.edge_index.shape}")

    def initialize_model(self):
        """初始化模型"""
        print("Initializing variational Bayesian GCN model...")

        self.gcn = GCN(
            device=self.device,
            user_num=self.user_num,
            item_num=self.item_num,
            embedding_dim=self.config.get('embedding_dim', 64),
            hidden_dim=self.config.get('hidden_dim', 128),
            num_layers=self.config.get('num_layers', 2)
        ).to(self.device)

        # 设置是否使用元学习
        self.gcn.fusion_net.set_meta_learning_mode(
            self.config.get('use_meta_learning', False)
        )

        print(f"Model initialized with {sum(p.numel() for p in self.gcn.parameters())} parameters")

    def initialize_optimizer(self):
        """初始化优化器和学习率调度器"""
        print("Initializing optimizer...")

        # 分组参数：不同组件使用不同学习率
        param_groups = [
            {
                'params': self.gcn.fusion_net.parameters(),
                'lr': self.config.get('fusion_lr', 1e-3),
                'weight_decay': self.config.get('fusion_weight_decay', 1e-5)
            },
            {
                'params': [p for name, p in self.gcn.named_parameters()
                           if 'fusion_net' not in name],
                'lr': self.config.get('graph_lr', 1e-3),
                'weight_decay': self.config.get('graph_weight_decay', 1e-6)
            }
        ]

        self.optimizer = torch.optim.AdamW(param_groups)

        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.gcn.train()

        epoch_losses = {
            'total': 0.0,
            'bpr': 0.0,
            'fusion': 0.0,
            'graph': 0.0,
            'kl': 0.0,
            'reconstruction': 0.0
        }

        num_batches = len(self.train_loader)

        for step, (batch_i, batch_j, batch_m) in enumerate(self.train_loader):
            batch_i = batch_i.to(self.device)
            batch_j = batch_j.to(self.device)
            batch_m = batch_m.to(self.device)

            self.optimizer.zero_grad()

            # 前向传播
            embeddings, loss_dict = self.gcn(
                self.video_features, self.audio_features,
                self.title_features, self.edge_index
            )

            # BPR损失计算
            user_embeddings = embeddings[batch_i]
            pos_item_embeddings = embeddings[batch_j + self.user_num]
            neg_item_embeddings = embeddings[batch_m + self.user_num]

            pos_scores = torch.sum(user_embeddings * pos_item_embeddings, dim=1)
            neg_scores = torch.sum(user_embeddings * neg_item_embeddings, dim=1)

            bpr_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10))

            # 确保fusion_loss和graph_loss是标量
            fusion_loss = loss_dict.get('fusion_loss', torch.tensor(0.0, device=self.device))
            graph_loss = loss_dict.get('graph_loss', torch.tensor(0.0, device=self.device))

            if isinstance(fusion_loss, torch.Tensor):
                fusion_loss = fusion_loss.item() if fusion_loss.numel() == 1 else torch.mean(fusion_loss).item()
            if isinstance(graph_loss, torch.Tensor):
                graph_loss = graph_loss.item() if graph_loss.numel() == 1 else torch.mean(graph_loss).item()

            # 总损失
            total_loss = bpr_loss + fusion_loss + graph_loss

            # 检查损失是否有效
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"Warning: Invalid loss detected at step {step}, skipping...")
                continue

            # 反向传播
            total_loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.gcn.parameters(), max_norm=1.0)

            self.optimizer.step()

            # 记录损失 - 确保转换为Python数值
            epoch_losses['total'] += total_loss.item()
            epoch_losses['bpr'] += bpr_loss.item()
            epoch_losses['fusion'] += fusion_loss
            epoch_losses['graph'] += graph_loss

            # 打印进度
            if step % self.config.get('print_freq', 100) == 0:
                print(f'Epoch [{epoch + 1}/{self.config["num_epochs"]}], '
                      f'Step [{step + 1}/{num_batches}], '
                      f'Loss: {total_loss.item():.4f}, '
                      f'BPR: {bpr_loss.item():.4f}, '
                      f'Fusion: {fusion_loss:.4f}')

        # 计算平均损失
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        return epoch_losses

    def evaluate_model(self):
        """评估模型性能"""
        self.gcn.eval()

        with torch.no_grad():
            # 获取所有嵌入
            embeddings = self.gcn(
                self.video_features, self.audio_features,
                self.title_features, self.edge_index
            )

            if isinstance(embeddings, tuple):
                embeddings = embeddings[0]

            # 计算评估指标（简化版本）
            user_embeddings = embeddings[:self.user_num]
            item_embeddings = embeddings[self.user_num:]

            # 计算相似度矩阵
            similarity_matrix = torch.mm(user_embeddings, item_embeddings.t())

            # 计算平均嵌入范数（作为质量指标）
            avg_user_norm = torch.mean(torch.norm(user_embeddings, dim=1))
            avg_item_norm = torch.mean(torch.norm(item_embeddings, dim=1))

            eval_metrics = {
                'avg_user_embedding_norm': avg_user_norm.item(),
                'avg_item_embedding_norm': avg_item_norm.item(),
                'similarity_range': (similarity_matrix.min().item(), similarity_matrix.max().item())
            }

        return eval_metrics

    def analyze_fusion_patterns(self, epoch):
        """分析融合模式"""
        if epoch % self.config.get('analysis_freq', 10) == 0:
            print(f"Analyzing fusion patterns at epoch {epoch}...")

            analysis_results = self.gcn.analyze_fusion_patterns(
                self.video_features, self.audio_features, self.title_features,
                num_samples=50
            )

            # 打印分析结果
            modality_importance = analysis_results['modality_importance']
            print(f"Average modality importance:")
            print(f"  Video: {np.mean(modality_importance['video']):.3f}")
            print(f"  Audio: {np.mean(modality_importance['audio']):.3f}")
            print(f"  Title: {np.mean(modality_importance['title']):.3f}")
            print(f"Average uncertainty: {np.mean(analysis_results['uncertainty_scores']):.3f}")

            return analysis_results

        return None

    def save_checkpoint(self, epoch, loss):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.gcn.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'config': self.config,
            'dataset_name': self.dataset_name,
            'train_stats': self.train_stats
        }

        checkpoint_dir = f'./checkpoints/{self.dataset_name}'
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = f'{checkpoint_dir}/checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    def train(self):
        """主训练循环"""
        print(f"Starting variational Bayesian training for {self.dataset_name}...")

        best_loss = float('inf')
        patience = self.config.get('patience', 10)
        patience_counter = 0

        for epoch in range(self.config['num_epochs']):
            print(f"\n=== Epoch {epoch + 1}/{self.config['num_epochs']} ===")

            # 训练一个epoch
            epoch_losses = self.train_epoch(epoch)

            # 记录统计信息
            self.train_stats['total_losses'].append(float(epoch_losses['total']))
            self.train_stats['fusion_losses'].append(float(epoch_losses['fusion']))
            self.train_stats['graph_losses'].append(float(epoch_losses['graph']))

            # 学习率调度
            self.scheduler.step(epoch_losses['total'])

            # 评估模型
            eval_metrics = self.evaluate_model()

            # 分析融合模式
            fusion_analysis = self.analyze_fusion_patterns(epoch)

            # 打印结果
            print(f"Epoch {epoch + 1} Results:")
            print(f"  Total Loss: {epoch_losses['total']:.4f}")
            print(f"  BPR Loss: {epoch_losses['bpr']:.4f}")
            print(f"  Fusion Loss: {epoch_losses['fusion']:.4f}")
            print(f"  Graph Loss: {epoch_losses['graph']:.4f}")
            print(f"  Avg User Norm: {eval_metrics['avg_user_embedding_norm']:.4f}")
            print(f"  Avg Item Norm: {eval_metrics['avg_item_embedding_norm']:.4f}")

            # 早停检查
            current_loss = epoch_losses['total']
            if current_loss < best_loss and not (
                    torch.isnan(torch.tensor(current_loss)) or torch.isinf(torch.tensor(current_loss))):
                best_loss = current_loss
                patience_counter = 0

                # 保存最佳模型
                self.save_checkpoint(epoch, current_loss)

            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        print("Training completed!")

        # 保存最终嵌入
        self.save_final_embeddings()

        # 生成训练报告
        self.generate_training_report()

    def save_final_embeddings(self):
        """保存最终的嵌入向量"""
        print("Saving final embeddings...")

        self.gcn.eval()
        with torch.no_grad():
            final_embeddings = self.gcn(
                self.video_features, self.audio_features,
                self.title_features, self.edge_index
            )

            if isinstance(final_embeddings, tuple):
                final_embeddings = final_embeddings[0]

            final_embeddings = final_embeddings.cpu().numpy()

        # 为每个数据集创建单独的输出目录
        output_dir = f'./output/{self.dataset_name}'
        os.makedirs(output_dir, exist_ok=True)

        np.save(f'{output_dir}/variational_bayesian_embeddings.npy', final_embeddings)

        # 分别保存用户和物品嵌入
        user_embeddings = final_embeddings[:self.user_num]
        item_embeddings = final_embeddings[self.user_num:]

        np.save(f'{output_dir}/user_embeddings.npy', user_embeddings)
        np.save(f'{output_dir}/item_embeddings.npy', item_embeddings)

        print(f"Embeddings saved to {output_dir} with shape: {final_embeddings.shape}")

    def generate_training_report(self):
        """生成训练报告"""
        print("Generating training report...")

        # 确保所有数据都是Python数值
        total_losses = [float(x) for x in self.train_stats['total_losses']]
        fusion_losses = [float(x) for x in self.train_stats['fusion_losses']]
        graph_losses = [float(x) for x in self.train_stats['graph_losses']]

        # 设置matplotlib后端
        import matplotlib
        matplotlib.use('Agg')  # 使用非交互式后端
        import matplotlib.pyplot as plt

        # 绘制损失曲线
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.plot(total_losses, label='Total Loss')
        plt.title(f'Total Training Loss - {self.dataset_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 3, 2)
        plt.plot(fusion_losses, label='Fusion Loss', color='orange')
        plt.title(f'Fusion Loss - {self.dataset_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 3, 3)
        plt.plot(graph_losses, label='Graph Loss', color='green')
        plt.title(f'Graph Loss - {self.dataset_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        output_dir = f'./output/{self.dataset_name}'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f'{output_dir}/training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Training report generated and saved to {output_dir}/training_curves.png")


def main():
    """主函数"""
    # 训练配置
    config = {
        'batch_size': 256,
        'num_epochs': 50,
        'embedding_dim': 64,
        'hidden_dim': 128,
        'num_layers': 2,
        'fusion_lr': 1e-3,
        'graph_lr': 1e-3,
        'fusion_weight_decay': 1e-5,
        'graph_weight_decay': 1e-6,
        'print_freq': 50,
        'analysis_freq': 5,
        'patience': 10,
        'use_meta_learning': False,  # 可以设置为True启用元学习
        'data_blocks': 1
    }

    # 训练所有数据集
    datasets = ['movielens', 'tiktok', 'Kwai']

    for dataset in datasets:
        print(f"\n{'=' * 60}")
        print(f"Training on dataset: {dataset}")
        print(f"{'=' * 60}")

        try:
            # 初始化训练器
            trainer = VariationalBayesianTrainer(config, dataset_name=dataset)

            # 开始训练
            trainer.train()

            print(f"Training completed for {dataset}")

        except Exception as e:
            print(f"Error training {dataset}: {e}")
            continue


if __name__ == "__main__":
    main()