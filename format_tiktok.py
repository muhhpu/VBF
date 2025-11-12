import json
import numpy as np
import os


def quick_explore_tiktok():
    """
    快速探索TikTok数据结构
    """
    data_dir = './dataset/tiktok'

    print("TikTok数据快速探索:")
    print("=" * 50)

    # 1. 检查JSON文件的第一行
    json_files = ['train.json', 'test.json', 'val.json']

    for json_file in json_files:
        file_path = os.path.join(data_dir, json_file)
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()

            print(f"\n{json_file} 第一行:")
            print(f"  内容: {first_line}")

            # 尝试解析
            try:
                data = json.loads(first_line)
                print(f"  解析成功: {data}")
                print(f"  数据类型: {type(data)}")
                if isinstance(data, dict):
                    print(f"  字段: {list(data.keys())}")
            except:
                print(f"  JSON解析失败")

    # 2. 检查特征文件基本信息
    feature_files = ['audio_feat.npy', 'image_feat.npy', 'text_feat.npy']

    print(f"\n特征文件:")
    for feat_file in feature_files:
        file_path = os.path.join(data_dir, feat_file)
        if os.path.exists(file_path):
            try:
                feat_data = np.load(file_path)
                print(f"  {feat_file}: 形状={feat_data.shape}, 类型={feat_data.dtype}")
            except Exception as e:
                print(f"  {feat_file}: 加载失败 - {e}")

    # 3. 检查train_mat
    train_mat_path = os.path.join(data_dir, 'train_mat')
    if os.path.exists(train_mat_path):
        print(f"\n发现train_mat文件，大小: {os.path.getsize(train_mat_path) / 1024:.1f} KB")

        # 尝试读取前几个字节
        with open(train_mat_path, 'rb') as f:
            first_bytes = f.read(100)
        print(f"  前100字节: {first_bytes}")


if __name__ == "__main__":
    quick_explore_tiktok()