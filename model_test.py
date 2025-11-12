import math
import numpy as np
import pickle
import torch
import os


def test_model(dataset_name='movielens'):
    """测试指定数据集的模型"""
    print(f"Testing model for dataset: {dataset_name}")

    # 加载数据参数
    para_file = f'./pro_data/{dataset_name}_load.para'
    if not os.path.exists(para_file):
        print(f"Parameter file {para_file} not found!")
        return

    with open(para_file, 'rb') as f_1:
        para_load = pickle.load(f_1)

    user_num = para_load['user_num']  # total number of users
    item_num = para_load['item_num']  # total number of items

    train_matrix = para_load['train_matrix']
    train_matrix.data = np.array(train_matrix.data, dtype=np.int8)
    train_matrix = train_matrix.toarray()  # the 0-1 matrix of training set

    test_matrix = para_load['test_matrix']
    test_matrix.data = np.array(test_matrix.data, dtype=np.int8)
    test_matrix = test_matrix.toarray()  # the 0-1 matrix of testing set

    print(f"Dataset: {dataset_name}")
    print(f"Users: {user_num}, Items: {item_num}")
    print(f"Train interactions: {np.sum(train_matrix)}")
    print(f"Test interactions: {np.sum(test_matrix)}")

    # 加载嵌入向量
    embedding_file = f'./output/{dataset_name}/variational_bayesian_embeddings.npy'
    if not os.path.exists(embedding_file):
        print(f"Embedding file {embedding_file} not found!")
        return

    x = np.load(embedding_file)
    print(f"Loaded embeddings with shape: {x.shape}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    item_ids = np.array(list(range(item_num)))

    user_matrix = x[: user_num, :]
    item_matrix = x[user_num: user_num + item_num, :]

    P = 0
    R = 0
    HR = 0
    NDCG = 0
    eva_size = 10

    def IDCG(num):
        if num == 0:
            return 1
        idcg = 0
        for i in list(range(num)):
            idcg += 1 / math.log(i + 2, 2)
        return idcg

    def descend_sort(array):
        return -np.sort(-array)

    null_user = 0

    for user_id, row in enumerate(test_matrix):
        if row.sum() == 0:
            null_user += 1
            continue
        can_item_ids = item_ids[~np.array(train_matrix[user_id], dtype=bool)]  # the id list of test items

        if len(can_item_ids) == 0:
            null_user += 1
            continue

        I = item_matrix[can_item_ids, :]
        u = user_matrix[user_id, :]
        inner_pro = np.matmul(u, I.T).reshape(-1)
        sort_index = np.argsort(-inner_pro)
        hit_num = 0
        dcg = 0
        for i, item_id in enumerate(can_item_ids[sort_index][0:eva_size]):
            if row[item_id] > 0:
                hit_num = hit_num + 1
                dcg = dcg + 1 / math.log(i + 2, 2)
        P += hit_num / eva_size
        R += hit_num / np.sum(row)
        HR += hit_num
        NDCG += dcg / IDCG(np.sum(descend_sort(row)[0:eva_size]))

    valid_users = user_num - null_user
    if valid_users > 0:
        P = P / valid_users
        R = R / valid_users
        HR = HR / np.sum(test_matrix)
        NDCG = NDCG / valid_users

        print(f'\nResults for {dataset_name}:')
        print(f'Valid users: {valid_users}/{user_num}')
        print(f'P@{eva_size}: {P:.4f}; R@{eva_size}: {R:.4f}; HR@{eva_size}: {HR:.4f}; NDCG@{eva_size}: {NDCG:.4f}')

        # 保存结果
        results = {
            'dataset': dataset_name,
            'precision': P,
            'recall': R,
            'hr': HR,
            'ndcg': NDCG,
            'valid_users': valid_users,
            'total_users': user_num
        }

        result_dir = f'./output/{dataset_name}'
        os.makedirs(result_dir, exist_ok=True)
        np.save(f'{result_dir}/test_results.npy', results)

        return results
    else:
        print(f"No valid users found for testing in {dataset_name}")
        return None


def main():
    """测试所有数据集"""
    datasets = ['movielens', 'tiktok', 'Kwai']
    all_results = []

    print("Testing all datasets...")
    print("=" * 60)

    for dataset in datasets:
        try:
            results = test_model(dataset)
            if results:
                all_results.append(results)
            print("-" * 60)
        except Exception as e:
            print(f"Error testing {dataset}: {e}")
            print("-" * 60)

    # 打印总结
    if all_results:
        print("\nSUMMARY OF ALL RESULTS:")
        print("=" * 60)
        for result in all_results:
            print(f"{result['dataset']}:")
            print(f"  P@10: {result['precision']:.4f}")
            print(f"  R@10: {result['recall']:.4f}")
            print(f"  HR@10: {result['hr']:.4f}")
            print(f"  NDCG@10: {result['ndcg']:.4f}")
            print(f"  Valid users: {result['valid_users']}/{result['total_users']}")
            print()


if __name__ == "__main__":
    # 可以单独测试某个数据集
    # test_model('movielens')
    # test_model('tiktok')
    # test_model('Kwai')

    # 或者测试所有数据集
    main()