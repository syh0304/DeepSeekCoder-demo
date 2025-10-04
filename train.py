from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from config import Config
import torch
from datasets import load_dataset

def train():
    # 加载配置
    cfg = Config()
    
    # 加载tokenizer和模型
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # 加载数据集
    dataset = load_dataset('json', data_files={
        'train': cfg.train_data_path,
        'validation': cfg.eval_data_path
    })
    
    # 数据预处理
    def tokenize_function(examples):
        return tokenizer(examples["code"], truncation=True, max_length=cfg.max_seq_length)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=cfg.batch_size,
        num_train_epochs=cfg.num_epochs,
        fp16=True,
        save_steps=1000,
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=500,
        gradient_accumulation_steps=4,
    )
    
    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
    )
    
    # 开始训练
    trainer.train()
    trainer.save_model("./fine_tuned_model")

if __name__ == "__main__":
    train()
