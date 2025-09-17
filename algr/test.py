from datasets import load_dataset
data_file = "/home/admin/.cache/huggingface/modules/datasets_modules/datasets/AL-GR--AL-GR-Tiny/25dea07242891a2d/train_data/s1_tiny.csv"
dataset = load_dataset("csv", data_files=data_file)
print(dataset)