import pandas as pd
import numpy as np
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import pickle
from tqdm import tqdm


def process_movielens_full_data():
    """
    处理完整的MovieLens数据，生成所需的.npy文件
    """
    print("开始处理完整MovieLens数据...")

    # 数据路径
    data_dir = './dataset/ml-10M100K'
    output_dir = './dataset_sample/movielens'

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 1. 读取完整评分数据
    print("1. 读取完整评分数据...")

    ratings_file = os.path.join(data_dir, 'ratings.dat')
    print(f"读取评分数据: {ratings_file}")

    # 使用更高效的方式读取大文件
    ratings_data = []
    with open(ratings_file, 'r', encoding='latin-1') as f:
        for line_num, line in enumerate(tqdm(f, desc="读取评分")):
            parts = line.strip().split('::')
            user_id = int(parts[0])
            movie_id = int(parts[1])
            rating = float(parts[2])
            timestamp = int(parts[3])
            ratings_data.append([user_id, movie_id, rating, timestamp])

            # 每100万行打印一次进度
            if (line_num + 1) % 1000000 == 0:
                print(f"已读取 {line_num + 1} 行")

    ratings_df = pd.DataFrame(ratings_data, columns=['user_id', 'movie_id', 'rating', 'timestamp'])
    print(f"总评分数: {len(ratings_df):,}")
    print(f"用户数: {ratings_df['user_id'].nunique():,}")
    print(f"电影数: {ratings_df['movie_id'].nunique():,}")
    print(f"评分范围: {ratings_df['rating'].min()} - {ratings_df['rating'].max()}")

    # 2. 读取完整电影信息
    print("\n2. 读取完整电影信息...")

    movies_file = os.path.join(data_dir, 'movies.dat')
    movies_data = []
    with open(movies_file, 'r', encoding='latin-1') as f:
        for line in tqdm(f, desc="读取电影信息"):
            parts = line.strip().split('::')
            movie_id = int(parts[0])
            title = parts[1]
            genres = parts[2]
            movies_data.append([movie_id, title, genres])

    movies_df = pd.DataFrame(movies_data, columns=['movie_id', 'title', 'genres'])
    print(f"电影信息数: {len(movies_df):,}")

    # 3. 读取标签数据（如果存在）
    print("\n3. 读取标签数据...")

    tags_file = os.path.join(data_dir, 'tags.dat')
    tags_df = None
    if os.path.exists(tags_file):
        print(f"读取标签数据: {tags_file}")
        tags_data = []
        with open(tags_file, 'r', encoding='latin-1') as f:
            for line in tqdm(f, desc="读取标签"):
                parts = line.strip().split('::')
                user_id = int(parts[0])
                movie_id = int(parts[1])
                tag = parts[2]
                timestamp = int(parts[3])
                tags_data.append([user_id, movie_id, tag, timestamp])

        tags_df = pd.DataFrame(tags_data, columns=['user_id', 'movie_id', 'tag', 'timestamp'])
        print(f"标签数: {len(tags_df):,}")

    # 4. 数据预处理 - 只保留隐式正反馈
    print("\n4. 数据预处理...")

    # 只保留评分>=4的正反馈（隐式反馈）
    print("过滤正反馈 (rating >= 4.0)...")
    positive_ratings = ratings_df[ratings_df['rating'] >= 4.0].copy()
    print(
        f"正反馈数量: {len(positive_ratings):,} / {len(ratings_df):,} ({len(positive_ratings) / len(ratings_df) * 100:.1f}%)")

    # 统计交互频次
    print("统计用户和物品交互频次...")
    user_counts = positive_ratings['user_id'].value_counts()
    item_counts = positive_ratings['movie_id'].value_counts()

    print(f"用户交互次数: 最小={user_counts.min()}, 最大={user_counts.max()}, 平均={user_counts.mean():.1f}")
    print(f"物品交互次数: 最小={item_counts.min()}, 最大={item_counts.max()}, 平均={item_counts.mean():.1f}")

    # 可选：过滤极低频用户和物品（至少5次交互）
    min_interactions = 5
    print(f"过滤少于{min_interactions}次交互的用户和物品...")

    valid_users = user_counts[user_counts >= min_interactions].index
    valid_items = item_counts[item_counts >= min_interactions].index

    filtered_ratings = positive_ratings[
        (positive_ratings['user_id'].isin(valid_users)) &
        (positive_ratings['movie_id'].isin(valid_items))
        ].copy()

    print(f"过滤前: {positive_ratings['user_id'].nunique():,} 用户, {positive_ratings['movie_id'].nunique():,} 物品")
    print(f"过滤后: {filtered_ratings['user_id'].nunique():,} 用户, {filtered_ratings['movie_id'].nunique():,} 物品")
    print(f"过滤后交互数: {len(filtered_ratings):,}")

    # 5. 重新编码ID（从0开始的连续ID）
    print("\n5. 重新编码用户和物品ID...")

    # 用户ID映射
    unique_users = sorted(filtered_ratings['user_id'].unique())
    user_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_users)}
    user_id_reverse_map = {new_id: old_id for old_id, new_id in user_id_map.items()}

    # 物品ID映射
    unique_items = sorted(filtered_ratings['movie_id'].unique())
    item_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_items)}
    item_id_reverse_map = {new_id: old_id for old_id, new_id in item_id_map.items()}

    # 应用映射
    filtered_ratings['user_new_id'] = filtered_ratings['user_id'].map(user_id_map)
    filtered_ratings['item_new_id'] = filtered_ratings['movie_id'].map(item_id_map)

    print(f"用户ID: {len(unique_users):,} 个, 范围 0 - {len(unique_users) - 1}")
    print(f"物品ID: {len(unique_items):,} 个, 范围 0 - {len(unique_items) - 1}")

    # 6. 生成用户-物品交互字典
    print("\n6. 生成用户-物品交互字典...")

    user_item_dict = {}
    grouped = filtered_ratings.groupby('user_new_id')['item_new_id'].apply(list)

    for user_new_id in tqdm(range(len(unique_users)), desc="构建用户字典"):
        if user_new_id in grouped.index:
            user_item_dict[user_new_id] = grouped[user_new_id]
        else:
            user_item_dict[user_new_id] = []

    # 保存用户-物品字典
    dict_path = os.path.join(output_dir, 'user_item_dict_sample.npy')
    np.save(dict_path, user_item_dict)
    print(f"保存用户-物品字典到: {dict_path}")

    # 统计信息
    interaction_counts = [len(items) for items in user_item_dict.values()]
    print(
        f"用户交互统计: 最小={min(interaction_counts)}, 最大={max(interaction_counts)}, 平均={np.mean(interaction_counts):.1f}")

    # 7. 数据分割（训练/测试/验证）
    print("\n7. 数据分割...")

    train_data = []
    test_data = []
    val_data = []

    # 设置分割比例
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1

    for user_new_id in tqdm(range(len(unique_users)), desc="分割数据"):
        items = user_item_dict[user_new_id]

        if len(items) == 0:
            # 无交互的用户
            test_data.append([user_new_id])
            val_data.append([user_new_id])
            continue
        elif len(items) == 1:
            # 只有一个交互的用户，全部给训练集
            train_data.append([user_new_id, items[0]])
            test_data.append([user_new_id])
            val_data.append([user_new_id])
            continue
        elif len(items) == 2:
            # 两个交互：1个训练，1个测试
            train_data.append([user_new_id, items[0]])
            test_data.append([user_new_id, items[1]])
            val_data.append([user_new_id])
            continue

        # 多个交互：随机分割
        items_copy = items.copy()
        random.shuffle(items_copy)

        n_items = len(items_copy)
        n_train = max(1, int(train_ratio * n_items))
        n_val = max(1, int(val_ratio * n_items))
        n_test = max(1, n_items - n_train - n_val)

        # 确保分割合理
        if n_train + n_val + n_test > n_items:
            if n_val > 1:
                n_val -= 1
            elif n_test > 1:
                n_test -= 1

        train_items = items_copy[:n_train]
        val_items = items_copy[n_train:n_train + n_val]
        test_items = items_copy[n_train + n_val:n_train + n_val + n_test]

        # 添加到训练集
        for item in train_items:
            train_data.append([user_new_id, item])

        # 添加到测试集和验证集
        if len(test_items) > 0:
            test_data.append([user_new_id] + test_items)
        else:
            test_data.append([user_new_id])

        if len(val_items) > 0:
            val_data.append([user_new_id] + val_items)
        else:
            val_data.append([user_new_id])

    # 转换为numpy数组并保存
    train_array = np.array(train_data, dtype=int)
    test_array = np.array(test_data, dtype=object)
    val_array = np.array(val_data, dtype=object)

    train_path = os.path.join(output_dir, 'train_sample.npy')
    test_path = os.path.join(output_dir, 'test_sample.npy')
    val_path = os.path.join(output_dir, 'val_sample.npy')

    np.save(train_path, train_array)
    np.save(test_path, test_array)
    np.save(val_path, val_array)

    print(f"保存训练集到: {train_path}, 形状: {train_array.shape}")
    print(f"保存测试集到: {test_path}, 形状: {test_array.shape}")
    print(f"保存验证集到: {val_path}, 形状: {val_array.shape}")

    # 8. 生成特征数据
    print("\n8. 生成特征数据...")

    n_items = len(unique_items)
    print(f"为 {n_items:,} 个物品生成特征...")

    # 8.1 生成视频特征 (2048维) - 模拟ResNet特征
    print("生成视频特征 (2048维)...")
    np.random.seed(42)
    # 使用正态分布，模拟ResNet输出
    v_features = np.random.normal(0, 1, (n_items, 2048)).astype(np.float64)
    # 添加一些稀疏性（模拟ReLU激活）
    v_features = np.maximum(0, v_features)
    # 标准化
    scaler = StandardScaler()
    v_features = scaler.fit_transform(v_features)

    v_feat_path = os.path.join(output_dir, 'v_feat_sample.npy')
    np.save(v_feat_path, v_features)
    print(f"保存视频特征到: {v_feat_path}, 形状: {v_features.shape}")

    # 8.2 生成音频特征 (128维) - 模拟VGGish特征
    print("生成音频特征 (128维)...")
    # VGGish输出通常经过量化，范围较大
    a_features = np.random.normal(0, 1, (n_items, 128)).astype(np.float64)
    # 添加一些音频特征的特性
    a_features = a_features * 2  # 增加方差
    a_features = np.tanh(a_features) * 10  # 限制范围但允许较大值

    a_feat_path = os.path.join(output_dir, 'a_feat_sample.npy')
    np.save(a_feat_path, a_features)
    print(f"保存音频特征到: {a_feat_path}, 形状: {a_features.shape}")

    # 8.3 生成基于内容的文本特征 (100维)
    print("生成基于内容的文本特征 (100维)...")

    # 获取对应的电影信息
    item_movies = movies_df[movies_df['movie_id'].isin(unique_items)].copy()
    item_movies = item_movies.set_index('movie_id').reindex(unique_items).reset_index()

    print(f"匹配的电影信息: {len(item_movies)} / {n_items}")

    # 处理类型特征
    print("处理电影类型...")
    all_genres = set()
    for genres_str in item_movies['genres'].dropna():
        if isinstance(genres_str, str):
            genres = genres_str.split('|')
            all_genres.update(genres)

    genre_list = sorted(list(all_genres))
    print(f"电影类型数量: {len(genre_list)}")

    # 创建类型特征矩阵
    genre_features = np.zeros((n_items, len(genre_list)))
    for i, (_, movie) in enumerate(item_movies.iterrows()):
        if pd.notna(movie['genres']):
            genres = movie['genres'].split('|')
            for genre in genres:
                if genre in genre_list:
                    genre_idx = genre_list.index(genre)
                    genre_features[i, genre_idx] = 1

    # 处理标题特征
    print("处理电影标题...")
    titles = []
    for title in item_movies['title']:
        if pd.notna(title):
            titles.append(str(title))
        else:
            titles.append("")

    # 使用TF-IDF处理标题
    vectorizer = TfidfVectorizer(max_features=50, stop_words='english', lowercase=True)
    try:
        title_features = vectorizer.fit_transform(titles).toarray()
    except:
        # 如果TF-IDF失败，使用随机特征
        title_features = np.random.normal(0, 0.1, (n_items, 50))

    print(f"类型特征形状: {genre_features.shape}")
    print(f"标题特征形状: {title_features.shape}")

    # 合并特征
    content_features = np.concatenate([genre_features, title_features], axis=1)

    # 调整到100维
    if content_features.shape[1] < 100:
        additional_dim = 100 - content_features.shape[1]
        additional_features = np.random.normal(0, 0.1, (n_items, additional_dim))
        content_features = np.concatenate([content_features, additional_features], axis=1)
    elif content_features.shape[1] > 100:
        content_features = content_features[:, :100]

    # 标准化文本特征
    scaler_t = StandardScaler()
    content_features = scaler_t.fit_transform(content_features)

    t_feat_path = os.path.join(output_dir, 't_feat_sample.npy')
    np.save(t_feat_path, content_features.astype(np.float64))
    print(f"保存文本特征到: {t_feat_path}, 形状: {content_features.shape}")

    # 9. 保存映射和元信息
    print("\n9. 保存映射信息...")

    mapping_info = {
        'user_id_map': user_id_map,
        'item_id_map': item_id_map,
        'user_id_reverse_map': user_id_reverse_map,
        'item_id_reverse_map': item_id_reverse_map,
        'n_users': len(unique_users),
        'n_items': len(unique_items),
        'n_train_interactions': len(train_array),
        'original_ratings': len(ratings_df),
        'positive_ratings': len(positive_ratings),
        'filtered_ratings': len(filtered_ratings),
        'genre_list': genre_list if 'genre_list' in locals() else [],
        'feature_stats': {
            'v_feat_shape': v_features.shape,
            'a_feat_shape': a_features.shape,
            't_feat_shape': content_features.shape,
        }
    }

    mapping_path = os.path.join(output_dir, 'mapping_info.pkl')
    with open(mapping_path, 'wb') as f:
        pickle.dump(mapping_info, f)

    print(f"保存映射信息到: {mapping_path}")

    # 10. 最终统计报告
    print("\n" + "=" * 80)
    print("完整数据处理完成！最终统计报告:")
    print("=" * 80)
    print(f"原始数据规模:")
    print(f"  - 总评分数: {len(ratings_df):,}")
    print(f"  - 原始用户数: {ratings_df['user_id'].nunique():,}")
    print(f"  - 原始物品数: {ratings_df['movie_id'].nunique():,}")
    print(f"")
    print(f"处理后数据规模:")
    print(f"  - 用户数: {len(unique_users):,}")
    print(f"  - 物品数: {len(unique_items):,}")
    print(f"  - 训练交互数: {len(train_array):,}")
    print(f"  - 正反馈比例: {len(positive_ratings) / len(ratings_df) * 100:.1f}%")
    print(f"")
    print(f"特征数据:")
    print(f"  - 视频特征: {v_features.shape} (float64)")
    print(f"  - 音频特征: {a_features.shape} (float64)")
    print(f"  - 文本特征: {content_features.shape} (float64)")
    print(f"")
    print(f"输出文件:")
    for filename in ['user_item_dict_sample.npy', 'train_sample.npy', 'test_sample.npy',
                     'val_sample.npy', 'v_feat_sample.npy', 'a_feat_sample.npy',
                     't_feat_sample.npy', 'mapping_info.pkl']:
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / 1024 / 1024
            print(f"  - {filename}: {size_mb:.1f} MB")
    print(f"")
    print(f"输出目录: {output_dir}")
    print("=" * 80)


def main():
    """
    主函数
    """
    # 设置随机种子确保可重现性
    random.seed(42)
    np.random.seed(42)

    print("🚀 开始处理完整MovieLens 10M数据集...")
    print("注意：这将处理约1000万条评分记录，可能需要几分钟时间。")

    # 检查数据文件是否存在
    data_dir = './dataset/ml-10M100K'
    required_files = ['ratings.dat', 'movies.dat']

    for filename in required_files:
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"❌ 错误: 找不到必需的文件 {filepath}")
            print(f"请确保MovieLens 10M数据集已正确解压到 {data_dir} 目录")
            return
        else:
            size_mb = os.path.getsize(filepath) / 1024 / 1024
            print(f"✅ 找到文件: {filename} ({size_mb:.1f} MB)")

    # 处理数据
    process_movielens_full_data()

    print("\n✅ 所有处理完成！")
    print("现在你拥有了与原始sample数据格式完全一致的完整数据集。")


if __name__ == "__main__":
    main()