import pandas as pd
import numpy as np
import os
import pickle


def convert_kuairec_to_kwai_sample():
    """
    Convert KuaiRec 2.0 dataset to Kwai sample format
    """

    # Create output directory if it doesn't exist
    os.makedirs("dataset_sample/Kwai", exist_ok=True)

    print("Loading KuaiRec 2.0 data...")

    # Load source data from KuaiRec 2.0
    print("Loading big matrix...")
    big_matrix = pd.read_csv("dataset/KuaiRec 2.0/data/big_matrix.csv")

    print("Loading small matrix...")
    small_matrix = pd.read_csv("dataset/KuaiRec 2.0/data/small_matrix.csv")

    print("Loading social network...")
    social_network = pd.read_csv("dataset/KuaiRec 2.0/data/social_network.csv")
    social_network["friend_list"] = social_network["friend_list"].map(eval)

    print("Loading item categories...")
    item_categories = pd.read_csv("dataset/KuaiRec 2.0/data/item_categories.csv")
    item_categories["feat"] = item_categories["feat"].map(eval)

    print("Loading user features...")
    user_features = pd.read_csv("dataset/KuaiRec 2.0/data/user_features.csv")

    print("Loading item daily features...")
    item_daily_features = pd.read_csv("dataset/KuaiRec 2.0/data/item_daily_features.csv")

    print("\nConverting to Kwai sample format...")

    # Sample data to create smaller datasets for the sample folder
    # Sample users (first 1000 users from big matrix)
    sample_users = sorted(big_matrix['user_id'].unique())[:1000]

    # Filter big matrix for sample users
    sampled_big_matrix = big_matrix[big_matrix['user_id'].isin(sample_users)].copy()

    # Get videos that appear in the sampled data
    sample_videos = sampled_big_matrix['video_id'].unique()

    # Filter small matrix for sample users and videos
    sampled_small_matrix = small_matrix[
        (small_matrix['user_id'].isin(sample_users)) &
        (small_matrix['video_id'].isin(sample_videos))
        ].copy()

    # Create user-item dictionary format
    user_item_dict = {}
    for user_id in sample_users:
        # 从big_matrix中获取该用户的所有交互视频
        user_videos = sampled_big_matrix[sampled_big_matrix['user_id'] == user_id]['video_id'].tolist()
        if user_videos:  # 只添加有交互的用户
            user_item_dict[user_id] = user_videos

    print(f"Created user-item dict with {len(user_item_dict)} users")

    # 重新映射用户ID为连续的0-based索引
    user_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(user_item_dict.keys()))}

    # 应用用户ID映射
    mapped_user_item_dict = {}
    for old_user_id, videos in user_item_dict.items():
        new_user_id = user_mapping[old_user_id]
        mapped_user_item_dict[new_user_id] = videos

    print(f"Remapped user IDs: {len(mapped_user_item_dict)} users (0-{len(mapped_user_item_dict) - 1})")

    # Save user-item dictionary as numpy file (compatible with the data_load.py)
    np.save("dataset_sample/Kwai/user_item_dict_sample.npy", mapped_user_item_dict)

    # Create video features based on categories
    print("Creating video features...")

    # 获取样本视频的类别信息
    sampled_item_categories = item_categories[
        item_categories['video_id'].isin(sample_videos)
    ].copy()

    # 创建视频特征矩阵
    # 获取所有出现的标签
    all_tags = set()
    for feat_list in sampled_item_categories['feat']:
        all_tags.update(feat_list)
    all_tags = sorted(list(all_tags))

    print(f"Found {len(all_tags)} unique tags: {all_tags}")

    # 创建one-hot编码的特征矩阵
    max_video_id = max(sample_videos)
    feature_dim = len(all_tags) + 5  # 添加一些额外的特征维度
    video_features = np.random.randn(max_video_id + 1, feature_dim).astype(np.float32)

    # 为有类别信息的视频设置特征
    for _, row in sampled_item_categories.iterrows():
        video_id = row['video_id']
        feat_list = row['feat']

        # 重置该视频的特征
        video_features[video_id] = np.zeros(feature_dim)

        # 设置对应标签的特征为1
        for tag in feat_list:
            if tag in all_tags:
                tag_idx = all_tags.index(tag)
                video_features[video_id][tag_idx] = 1.0

        # 添加一些随机噪声特征
        video_features[video_id][-5:] = np.random.randn(5) * 0.1

    # 保存视频特征为PyTorch格式
    import torch
    torch.save(torch.from_numpy(video_features), "dataset_sample/Kwai/v_feat_sample.pt")

    print(f"Created video features: shape {video_features.shape}")
    print(f"Video ID range: 0-{max_video_id}")

    # 保存其他辅助文件（可选）
    sampled_social_network = social_network[
        social_network['user_id'].isin(sample_users)
    ].copy()

    sampled_user_features = user_features[
        user_features['user_id'].isin(sample_users)
    ].copy()

    sampled_item_daily = item_daily_features[
        item_daily_features['video_id'].isin(sample_videos)
    ].copy()

    # 保存其他数据文件
    sampled_big_matrix.to_csv("dataset_sample/Kwai/big_matrix_sample.csv", index=False)
    sampled_small_matrix.to_csv("dataset_sample/Kwai/small_matrix_sample.csv", index=False)
    sampled_social_network.to_csv("dataset_sample/Kwai/social_network_sample.csv", index=False)
    sampled_item_categories.to_csv("dataset_sample/Kwai/item_categories_sample.csv", index=False)
    sampled_user_features.to_csv("dataset_sample/Kwai/user_features_sample.csv", index=False)
    sampled_item_daily.to_csv("dataset_sample/Kwai/item_daily_features_sample.csv", index=False)

    print("\nConversion completed!")
    print(f"Sample users: {len(mapped_user_item_dict)}")
    print(f"Sample videos: {len(sample_videos)}")
    print(f"Total interactions: {len(sampled_big_matrix)}")
    print(f"Video feature dimensions: {feature_dim}")

    print("\nFiles created in dataset_sample/Kwai/:")
    for file in sorted(os.listdir("dataset_sample/Kwai")):
        print(f"  - {file}")

    # 验证生成的文件
    print("\nVerifying generated files...")

    # 验证user_item_dict
    loaded_dict = np.load("dataset_sample/Kwai/user_item_dict_sample.npy", allow_pickle=True).item()
    print(f"Loaded user-item dict: {len(loaded_dict)} users")

    # 验证视频特征
    loaded_features = torch.load("dataset_sample/Kwai/v_feat_sample.pt")
    print(f"Loaded video features: {loaded_features.shape}")

    return mapped_user_item_dict, video_features


if __name__ == "__main__":
    convert_kuairec_to_kwai_sample()