import torch

class Config:
    # 模型配置
    model_name = "deepseek-ai/deepseek-coder-1.3b"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 训练配置 (针对8G显存优化)
    batch_size = 1
    learning_rate = 5e-5
    max_seq_length = 512
    num_epochs = 3
    
    # 推理配置
    generation_max_length = 256
    temperature = 0.7
    
    # 数据路径
    train_data_path = "data/train_data.jsonl"
    eval_data_path = "data/eval_data.jsonl"
