# prepare_data.py
import pandas as pd
import json

# datasets: 
df = pd.read_csv("./train_data.csv", sep='\t')  # replace your filename
print(df)

with open("qwen_finetune_data.jsonl", "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        f.write(json.dumps({
            "id": row["id"],
            "instruction": row["system"],
            "input": row["user"].strip(),
            "output": row["answer"].strip()
        }, ensure_ascii=False) + "\n")

print("✅ test data is generated: qwen_finetune_data.jsonl")
