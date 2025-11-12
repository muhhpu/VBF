import numpy as np
import pickle
import gc
import os


def generate_triples(dataset_name, ratio=5):
    """
    为指定数据集生成三元组训练数据
    """
    print(f"Generating triples for dataset: {dataset_name}")

    # 创建输出目录
    os.makedirs('./pro_triple', exist_ok=True)

    # 加载数据
    para_file = f'./pro_data/{dataset_name}_load.para'
    if not os.path.exists(para_file):
        print(f"Warning: Parameter file {para_file} does not exist, skipping...")
        return

    with open(para_file, 'rb') as f_para:
        para = pickle.load(f_para)

    user_num = para['user_num']  # total number of users
    item_num = para['item_num']  # total number of items
    train_matrix = para['train_matrix']
    train_ui = para['train_ui']

    print(f"Dataset: {dataset_name}")
    print(f"Users: {user_num}, Items: {item_num}")
    print(f"Training interactions: {len(train_ui)}")

    print('train triple started...')
    item_ids = np.array(list(range(item_num)))
    train_triple = np.empty(shape=[0, 3], dtype=int)
    mtx = np.array(train_matrix.todense())
    para_index = 0

    for lh, inter in enumerate(train_ui):
        user_id = inter[0]  # user_id is an 1-D numpy array
        bool_index = ~np.array(mtx[user_id, :], dtype=bool)
        can_item_ids = item_ids[bool_index]  # the id list of 0-value items

        # 确保有足够的负样本
        if len(can_item_ids) < ratio:
            if len(can_item_ids) == 0:
                continue  # 跳过没有负样本的情况
            # 如果负样本不够，就重复采样
            a1 = np.random.choice(can_item_ids, size=ratio, replace=True)
        else:
            a1 = np.random.choice(can_item_ids, size=ratio, replace=False)  # a1 is an 1-D numpy array

        inter = np.expand_dims(inter, axis=0)
        inter = np.repeat(inter, repeats=ratio, axis=0)  # size = ratio * 2
        a1 = np.expand_dims(a1, axis=1)
        triple = np.append(inter, a1, axis=1)  # size = ratio * 3
        train_triple = np.append(train_triple, triple, axis=0)

        if lh % 10000 == 9999:
            print('=====[%d] ten thousand completed=====' % ((lh + 1) / 10000))

        # 每处理10万条数据保存一次，避免内存溢出
        if lh % 1e5 == (1e5 - 1) and lh < len(train_ui) - 1:
            train_i = train_triple[:, 0]
            train_j = train_triple[:, 1]
            train_m = train_triple[:, 2]
            para = {}
            para['train_i'] = train_i
            para['train_j'] = train_j
            para['train_m'] = train_m
            pickle.dump(para, open(f'./pro_triple/{dataset_name}_triple_{para_index}.para', 'wb'))
            print(f'para{para_index} saved...')
            train_triple = np.empty(shape=[0, 3], dtype=int)
            para_index += 1
            del para
            gc.collect()

    # 保存剩余的数据
    train_i = train_triple[:, 0]
    train_j = train_triple[:, 1]
    train_m = train_triple[:, 2]
    para = {}
    para['train_i'] = train_i
    para['train_j'] = train_j
    para['train_m'] = train_m
    pickle.dump(para, open(f'./pro_triple/{dataset_name}_triple_{para_index}.para', 'wb'))
    print(f'data_triple finished for {dataset_name}...')
    print(f"Generated {para_index + 1} triple files")


def main():
    """
    为所有数据集生成三元组
    """
    datasets = ['movielens', 'tiktok', 'Kwai']

    for dataset in datasets:
        try:
            generate_triples(dataset, ratio=5)
            print("-" * 50)
        except Exception as e:
            print(f"Error generating triples for {dataset}: {e}")
            print("-" * 50)


if __name__ == "__main__":
    # 设置随机种子
    np.random.seed(42)

    generate_triples('movielens', ratio=5)
    generate_triples('tiktok', ratio=5)
    generate_triples('Kwai', ratio=5)
