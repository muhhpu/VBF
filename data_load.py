import numpy as np
import random
import math
import scipy.sparse as sparse
import pickle
import torch
from sklearn.decomposition import PCA
import os


def process_dataset(dataset_name, has_v=True, has_a=True, has_t=True):
    """
    处理指定数据集
    """
    print(f"Processing dataset: {dataset_name}")

    # 根据数据集名称设置路径和参数
    if dataset_name == 'movielens':
        data_dir = './dataset_sample/movielens'
        output_dir = './pro_data'
        feature_dir = './pro_feature'
        user_item_dict_file = 'user_item_dict_sample.npy'
        v_feat_file = 'v_feat_sample.npy'
        a_feat_file = 'a_feat_sample.npy'
        t_feat_file = 't_feat_sample.npy'
        v_pca_dim = 64
        a_pca_dim = 64
        t_pca_dim = 32
    elif dataset_name == 'tiktok':  # 改为小写
        data_dir = './dataset_sample/tiktok'
        output_dir = './pro_data'
        feature_dir = './pro_feature'
        user_item_dict_file = 'user_item_dict_sample.npy'
        v_feat_file = 'v_feat_sample.pt'  # 修正文件名
        a_feat_file = 'a_feat_sample.pt'  # 修正文件名
        t_feat_file = 't_feat_sample.pt'  # 修正文件名
        v_pca_dim = 64
        a_pca_dim = 64
        t_pca_dim = 32
    elif dataset_name == 'Kwai':
        data_dir = './dataset_sample/Kwai'
        output_dir = './pro_data'
        feature_dir = './pro_feature'
        user_item_dict_file = 'user_item_dict_sample.npy'
        v_feat_file = 'v_feat_sample.pt'  # 修正文件名
        a_feat_file = None  # Kwai只有视频特征
        t_feat_file = None
        v_pca_dim = 64
        a_pca_dim = None
        t_pca_dim = None
        has_a = False
        has_t = False
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(feature_dir, exist_ok=True)

    # 检查数据目录是否存在
    if not os.path.exists(data_dir):
        print(f"Warning: Data directory {data_dir} does not exist, skipping...")
        return

    # 加载用户-物品字典
    user_item_dict_path = os.path.join(data_dir, user_item_dict_file)
    if not os.path.exists(user_item_dict_path):
        print(f"Warning: User-item dict file {user_item_dict_path} does not exist, skipping...")
        return

    whole_dict = np.load(user_item_dict_path, allow_pickle=True)
    whole_dict = whole_dict.item()

    # 获取所有出现的物品ID
    all_items = []
    for user_items in whole_dict.values():
        all_items.extend(user_items)
    all_items = sorted(list(set(all_items)))
    print(f"Found {len(all_items)} unique items in interactions, range: {min(all_items)}-{max(all_items)}")

    # 加载和处理特征数据
    video_signal_lookup = None
    audio_signal_lookup = None
    title_signal_lookup = None

    if has_v and v_feat_file:
        v_feat_path = os.path.join(data_dir, v_feat_file)
        if os.path.exists(v_feat_path):
            try:
                if v_feat_file.endswith('.npy'):
                    video_feature_lookup = np.load(v_feat_path)
                elif v_feat_file.endswith('.pt'):
                    video_feature_lookup = torch.load(v_feat_path, map_location='cpu')
                    if isinstance(video_feature_lookup, torch.Tensor):
                        video_feature_lookup = video_feature_lookup.numpy()

                print(f"Original video feature shape: {video_feature_lookup.shape}")

                # 检查特征数量是否足够
                if len(all_items) > video_feature_lookup.shape[0]:
                    print(
                        f"Warning: Not enough video features for all items. Features: {video_feature_lookup.shape[0]}, Items: {len(all_items)}")
                    # 只保留有特征的物品
                    valid_items = [item for item in all_items if item < video_feature_lookup.shape[0]]
                    all_items = valid_items
                    print(f"Using {len(all_items)} items with valid features")

                # 只保留交互数据中出现的物品的特征
                video_feature_lookup = video_feature_lookup[all_items]
                print(f"Filtered video feature shape: {video_feature_lookup.shape}")

                # PCA处理
                pca_v = PCA(n_components=min(v_pca_dim, video_feature_lookup.shape[1]))
                video_signal_lookup = pca_v.fit_transform(video_feature_lookup)

                # 保存特征
                v_output_path = os.path.join(feature_dir, f'{dataset_name}_v_{v_pca_dim}.npy')
                np.save(v_output_path, video_signal_lookup)
                print(f"Saved video features to: {v_output_path}")
            except Exception as e:
                print(f"Error loading video features: {e}")
                has_v = False
        else:
            print(f"Warning: Video feature file {v_feat_path} does not exist")
            has_v = False

    if has_a and a_feat_file:
        a_feat_path = os.path.join(data_dir, a_feat_file)
        if os.path.exists(a_feat_path):
            try:
                if a_feat_file.endswith('.npy'):
                    audio_feature_lookup = np.load(a_feat_path)
                elif a_feat_file.endswith('.pt'):
                    audio_feature_lookup = torch.load(a_feat_path, map_location='cpu')
                    if isinstance(audio_feature_lookup, torch.Tensor):
                        audio_feature_lookup = audio_feature_lookup.numpy()

                print(f"Original audio feature shape: {audio_feature_lookup.shape}")

                # 只保留交互数据中出现的物品的特征
                audio_feature_lookup = audio_feature_lookup[all_items]
                print(f"Filtered audio feature shape: {audio_feature_lookup.shape}")

                # PCA处理
                pca_a = PCA(n_components=min(a_pca_dim, audio_feature_lookup.shape[1]))
                audio_signal_lookup = pca_a.fit_transform(audio_feature_lookup)

                # 保存特征
                a_output_path = os.path.join(feature_dir, f'{dataset_name}_a_{a_pca_dim}.npy')
                np.save(a_output_path, audio_signal_lookup)
                print(f"Saved audio features to: {a_output_path}")
            except Exception as e:
                print(f"Error loading audio features: {e}")
                has_a = False
        else:
            print(f"Warning: Audio feature file {a_feat_path} does not exist")
            has_a = False

    if has_t and t_feat_file:
        t_feat_path = os.path.join(data_dir, t_feat_file)
        if os.path.exists(t_feat_path):
            try:
                if t_feat_file.endswith('.npy'):
                    title_feature_lookup = np.load(t_feat_path)
                elif t_feat_file.endswith('.pt'):
                    title_feature_lookup = torch.load(t_feat_path, map_location='cpu')
                    if isinstance(title_feature_lookup, torch.Tensor):
                        title_feature_lookup = title_feature_lookup.numpy()

                print(f"Original title feature shape: {title_feature_lookup.shape}")

                # 只保留交互数据中出现的物品的特征
                title_feature_lookup = title_feature_lookup[all_items]
                print(f"Filtered title feature shape: {title_feature_lookup.shape}")

                # PCA处理
                pca_t = PCA(n_components=min(t_pca_dim, title_feature_lookup.shape[1]))
                title_signal_lookup = pca_t.fit_transform(title_feature_lookup)

                # 保存特征
                t_output_path = os.path.join(feature_dir, f'{dataset_name}_t_{t_pca_dim}.npy')
                np.save(t_output_path, title_signal_lookup)
                print(f"Saved title features to: {t_output_path}")
            except Exception as e:
                print(f"Error loading title features: {e}")
                has_t = False
        else:
            print(f"Warning: Title feature file {t_feat_path} does not exist")
            has_t = False

    # 使用实际的用户数量和物品数量
    ACTUAL_USER_NUM = len(whole_dict)
    ACTUAL_ITEM_NUM = len(all_items)
    print(f"Using actual dimensions - Users: {ACTUAL_USER_NUM}, Items: {ACTUAL_ITEM_NUM}")

    # 创建物品ID映射（从原始ID映射到0-based连续索引）
    item_id_map = {old_id: new_id for new_id, old_id in enumerate(all_items)}

    # 初始化数据集
    train_ui = np.empty(shape=[0, 2], dtype=int)
    test_ui = np.empty(shape=[0, 2], dtype=int)
    val_ui = np.empty(shape=[0, 2], dtype=int)

    # 数据分割
    for k in range(ACTUAL_USER_NUM):
        if k not in whole_dict:
            continue

        whole_set = list(whole_dict[k])

        # 过滤掉不在有效物品列表中的物品
        whole_set = [item for item in whole_set if item in item_id_map]

        if len(whole_set) == 0:
            continue

        # 处理交互数据太少的情况
        if len(whole_set) < 3:
            for i in whole_set:
                mapped_item = item_id_map[i]
                train_ui = np.append(train_ui, [[k, mapped_item]], axis=0)
            continue

        train_set = random.sample(whole_set, max(1, round(0.8 * len(whole_set))))
        test_val_set = list(set(whole_set) - set(train_set))

        if len(test_val_set) == 0:
            test_set = [train_set.pop()]
            val_set = []
        elif len(test_val_set) == 1:
            test_set = test_val_set
            val_set = []
        else:
            test_set = random.sample(test_val_set, max(1, math.ceil(0.5 * len(test_val_set))))
            val_set = list(set(test_val_set) - set(test_set))

        # 使用映射后的物品ID
        for i in train_set:
            mapped_item = item_id_map[i]
            train_ui = np.append(train_ui, [[k, mapped_item]], axis=0)
        for j in test_set:
            mapped_item = item_id_map[j]
            test_ui = np.append(test_ui, [[k, mapped_item]], axis=0)
        for t in val_set:
            mapped_item = item_id_map[t]
            val_ui = np.append(val_ui, [[k, mapped_item]], axis=0)

    print('Creating sparse matrices...')

    # 使用实际的用户数和物品数创建稀疏矩阵
    row = train_ui[:, 0]
    col = train_ui[:, 1]
    data = np.ones_like(row)
    train_matrix = sparse.coo_matrix((data, (row, col)), shape=(ACTUAL_USER_NUM, ACTUAL_ITEM_NUM), dtype=np.int8)

    row = test_ui[:, 0]
    col = test_ui[:, 1]
    data = np.ones_like(row)
    test_matrix = sparse.coo_matrix((data, (row, col)), shape=(ACTUAL_USER_NUM, ACTUAL_ITEM_NUM), dtype=np.int8)

    row = val_ui[:, 0]
    col = val_ui[:, 1]
    data = np.ones_like(row)
    val_matrix = sparse.coo_matrix((data, (row, col)), shape=(ACTUAL_USER_NUM, ACTUAL_ITEM_NUM), dtype=np.int8)

    print('Saving processed data...')

    # 保存参数
    para = {}
    para['user_num'] = ACTUAL_USER_NUM
    para['item_num'] = ACTUAL_ITEM_NUM
    para['train_matrix'] = train_matrix
    para['test_matrix'] = test_matrix
    para['val_matrix'] = val_matrix
    para['train_ui'] = train_ui
    para['item_id_map'] = item_id_map
    para['has_v'] = has_v
    para['has_a'] = has_a
    para['has_t'] = has_t

    # 保存到对应数据集的文件
    para_output_path = os.path.join(output_dir, f'{dataset_name}_load.para')
    pickle.dump(para, open(para_output_path, 'wb'))

    print(f'Data processing finished for {dataset_name}...')
    print(f"Final dimensions - Users: {ACTUAL_USER_NUM}, Items: {ACTUAL_ITEM_NUM}")

    # 打印特征信息
    if has_v and video_signal_lookup is not None:
        print(f"Video features: {video_signal_lookup.shape}")
    if has_a and audio_signal_lookup is not None:
        print(f"Audio features: {audio_signal_lookup.shape}")
    if has_t and title_signal_lookup is not None:
        print(f"Title features: {title_signal_lookup.shape}")

    if len(train_ui) > 0:
        print(
            f"Max item index in interactions: {max(train_ui[:, 1].max(), test_ui[:, 1].max() if len(test_ui) > 0 else 0, val_ui[:, 1].max() if len(val_ui) > 0 else 0)}")
    print(f"Parameters saved to: {para_output_path}")
    print("-" * 50)


# 主函数：处理所有数据集
def main():
    """
    处理所有支持的数据集
    """
    # 处理 MovieLens 数据集
    try:
        process_dataset('movielens', has_v=True, has_a=True, has_t=True)
    except Exception as e:
        print(f"Error processing movielens: {e}")

    # 处理 TikTok 数据集
    try:
        process_dataset('tiktok', has_v=True, has_a=True, has_t=True)
    except Exception as e:
        print(f"Error processing tiktok: {e}")

    # 处理 Kwai 数据集（只有视频特征）
    try:
        process_dataset('Kwai', has_v=True, has_a=False, has_t=False)
    except Exception as e:
        print(f"Error processing Kwai: {e}")


if __name__ == "__main__":
    # 设置随机种子以确保结果可重现
    random.seed(42)
    np.random.seed(42)


    process_dataset('movielens', has_v=True, has_a=True, has_t=True)
    process_dataset('tiktok', has_v=True, has_a=True, has_t=False)
    process_dataset('Kwai', has_v=True, has_a=False, has_t=False)