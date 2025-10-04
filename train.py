import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

class DeepSeekCoderDemo:
    def __init__(self, model_name="deepseek-ai/deepseek-coder-1.3b"):
        """
        初始化DeepSeekCoder演示类
        使用1.3B版本以适应8G显存
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {self.device}")
        
        # 显存优化配置
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 根据显存容量选择合适的模型加载方式
        if torch.cuda.get_device_properties(0).total_memory < 9e9:  # 小于9GB
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16
            ).to(self.device)
            
        self.model.eval()
        print("模型加载完成!")
    
    def generate_code(self, prompt, max_length=512, temperature=0.7):
        """
        生成代码的函数
        
        Args:
            prompt: 输入提示
            max_length: 生成的最大长度
            temperature: 生成温度，控制随机性
        """
        # 编码输入
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # 生成参数设置
        generation_config = {
            "max_length": max_length,
            "temperature": temperature,
            "top_k": 50,
            "top_p": 0.9,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id
        }
        
        # 生成代码
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                **generation_config
            )
        
        # 解码输出
        generated_code = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_code

def check_gpu_memory():
    """检查GPU内存使用情况"""
    if torch.cuda.is_available():
        gpu_props = torch.cuda.get_device_properties(0)
        total_memory = gpu_props.total_memory / (1024 ** 3)  # 转换为GB
        allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
        
        print(f"GPU: {gpu_props.name}")
        print(f"总显存: {total_memory:.2f} GB")
        print(f"已分配: {allocated:.2f} GB")
        print(f"已保留: {reserved:.2f} GB")
    else:
        print("CUDA不可用")

if __name__ == "__main__":
    # 检查GPU状态
    check_gpu_memory()
    
    # 初始化模型
    coder = DeepSeekCoderDemo()
    
    # 示例提示
    prompt = """
    # 编写一个Python函数，计算斐波那契数列的第n项
    def fibonacci(n):
    """
    
    # 生成代码
    print("生成代码中...")
    result = coder.generate_code(prompt)
    print("生成的代码:")
    print(result)
