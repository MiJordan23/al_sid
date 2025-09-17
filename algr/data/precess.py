# prepare_data.py
import pandas as pd
import json

# datasets: 链接
df = pd.read_csv("./train_data.csv", sep='\t')  # 替换为你的文件路径
print(df)

with open("qwen_finetune_data.jsonl", "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        f.write(json.dumps({
            "id": row["id"],
            "instruction": row["system"],
            "input": row["user"].strip(),
            "output": row["answer"].strip()
        }, ensure_ascii=False) + "\n")

print("✅ 数据已生成: qwen_finetune_data.jsonl")
