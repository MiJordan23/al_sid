import json
import pandas as pd
import collections
import argparse
import os
from datasets import load_dataset
from tqdm import tqdm

def calculate_hit_rate_k(generate_text, answer, k_list, sid_to_ids_map):
    """
    计算单个样本的 HR@K
    """
    generated_ids = []
    for sid in generate_text:
        if sid in sid_to_ids_map:
            generated_ids.extend(sid_to_ids_map[sid])

    true_ids = set(answer.strip().split(";"))
    true_count = len(true_ids)
    hit_count = []
    for k in k_list.split(','):
        slice = min(len(generated_ids), int(k))
        gids = set(generated_ids[:slice])
        hit_count.append(len(gids & true_ids))
    return hit_count, true_count

def calculate_average_hit_rate_k(file_path, k_list, sid_to_ids_map, decoder_only=True):
    """
    计算所有样本的平均 HR@K
    """
    total_count = 0
    hr_count = [0] * len(k_list)
    with open(file_path, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            line = line.strip()
            if line:
                try:
                    sample = json.loads(line)
                    if decoder_only:
                        generate_text = sample["_generated_new_text_"]
                    else:
                        generate_text = sample["_generated_text_"]
                    answer = sample["answer"]
                    hit_count, true_count = calculate_hit_rate_k(generate_text, answer, k_list, sid_to_ids_map)
                    total_count += true_count
                    hr_count = [a + b for a, b in zip(hit_count, hr_count)]

                except json.JSONDecodeError as e:
                    print(f"Invalid JSON line: {line}")
                    print(f"Error: {e}")
    
    return [hrc / total_count if total_count > 0 else 0 for hrc in hr_count]


def convert_csv_to_map(data):
    """
    使用 Pandas 列操作 + groupby 优化性能
    参数:
        data: 原始数据（列表或 DataFrame）
    返回:
        sid_to_ids_map: 字典，键为 SID，值为对应的 item_id 列表
    """
    # 1. 转换为 DataFrame 并命名列
    df = pd.DataFrame(data).dropna()
    df.columns = ['item_id', 'codebook_lv1', 'codebook_lv2', 'codebook_lv3']  # 假设输入数据列为 [item_id, col1, col2, col3]

    # 2. 列操作：转换为整数并计算 num2, num3
    df['col1'] = df['codebook_lv1'].astype(int)
    df['col2'] = df['codebook_lv2'].astype(int) + 8192
    df['col3'] = df['codebook_lv3'].astype(int) + 8192 * 2

    # 3. 构建 sid 列
    df['sid'] = 'C' + df['col1'].astype(str) + 'C' + df['col2'].astype(str) + 'C' + df['col3'].astype(str)

    # 4. 按 sid 分组，聚合 item_id 为列表
    sid_to_ids_map = df.groupby('sid')['item_id'].agg(list).to_dict()
    return sid_to_ids_map


def main():
    parser = argparse.ArgumentParser()
    #'/home/admin/.cache/huggingface/modules/datasets_modules/datasets/AL-GR--AL-GR-Tiny/25dea07242891a2d'
    parser.add_argument('--dataset_name', type=str, default="AL-GR/AL-GR-Tiny")
    parser.add_argument('--item_sid_file', type=str, default="item_info/tiny_item_sid_final.csv")
    parser.add_argument('--generate_file', type=str, default="logs/generate_t5base_3layer_tiny/output.jsonl")
    parser.add_argument('--k_list', type=str, default="20,100,500,1000")
    parser.add_argument('--decoder_only', action="store_true")
    parser.add_argument('--nebula', action="store_true")

    args = parser.parse_args()
    dataset_name = args.dataset_name
    item_sid_file = args.item_sid_file
    file_path = args.generate_file
    # sid_to_ids_map = {
    #     "C3936C1881C5331": ["HiIRC", "xx", "xx"]
    #     # 其他 SID 映射...
    # }
    if args.nebula:
        item_sid_data = pd.read_csv(os.path.join(args.dataset_name, args.item_sid_file))
    else:
        item_sid_data = load_dataset(dataset_name, data_files=item_sid_file)
    
    ## 将处理好的文件写入到本地，下次运行的时候回快一些
    local_sid2item_file = f"./data/sid2item_v_{os.path.basename(item_sid_file).split('.')[0]}.json"
    if os.path.isfile(local_sid2item_file):
        # 读取 JSON 文件
        print("load file directly")
        with open(local_sid2item_file, "r", encoding="utf-8") as f:
            sid_to_ids_map = json.load(f)
        print("load data sucess!")
    else:
        print("process data firstly")
        sid_to_ids_map = convert_csv_to_map(item_sid_data)
        # 写入 JSON 文件
        with open(local_sid2item_file, "w", encoding="utf-8") as f:
            json.dump(sid_to_ids_map, f, indent=4, ensure_ascii=False)
        print("load data sucess!")

    average_hr = calculate_average_hit_rate_k(file_path, args.k_list, sid_to_ids_map, args.decoder_only)
    for k, hr in zip(args.k_list.split(','), average_hr):
        print(f"Average HR@{k}: {hr:.4f}")

if __name__ == '__main__':
    main()

    
    
    