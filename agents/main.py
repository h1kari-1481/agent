from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import os

# ============ 环境配置 ============
os.environ["OPENAI_API_KEY"] = "sk-346481cf0e964b67b50a538519b2c539"
os.environ["OPENAI_API_BASE"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# ============ 创建 DeepSeek 模型 ============
chat_llm = ChatOpenAI(
    model_name="deepseek-v3",
    temperature=0.3,
)

# ============ 提示模板（Prompt） ============
prompt = PromptTemplate(
    input_variables=["code"],
    template=(
        "你是一名资深代码审查专家。"
        "请仔细分析以下代码，并指出所有可能的Bug、逻辑错误、安全漏洞或性能问题。\n\n"
        "要求：\n"
        "1. 列出每个问题的代码位置（如行号或上下文）\n"
        "2. 简要说明问题原因\n"
        "3. 提出可行的修复建议\n\n"
        "以下是代码内容：\n\n{code}\n\n"
        "请用清晰、有条理的中文回答。"
    ),
)

# ============ 构建链 ============
chain = prompt | chat_llm

# ============ 读取待分析文件 ============
# 动态获取项目路径
import sys
import os
from pathlib import Path

# 获取项目根目录（Test目录）
project_root = Path(__file__).parent.parent / "Test"

# 查找主要的代码文件
code_files = list(project_root.glob("*.py")) + list(project_root.glob("*.js")) + list(project_root.glob("*.java"))

if code_files:
    # 使用找到的第一个代码文件
    file_path = str(code_files[0])
    print(f"分析文件: {file_path}")
else:
    # 如果没有找到代码文件，使用一个默认文件
    file_path = str(project_root / "example.py")
    print(f"未找到代码文件，将创建示例文件: {file_path}")

    # 创建示例文件用于测试
    example_code = '''# 示例Python代码
def calculate_average(numbers):
    """计算平均值"""
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)

def process_user_data(data):
    """处理用户数据"""
    result = []
    for item in data:
        if item and "name" in item:
            result.append(item["name"])
    return result

class DataManager:
    def __init__(self):
        self.data = []

    def add_data(self, new_data):
        self.data.extend(new_data)

    def get_summary(self):
        if not self.data:
            return {"count": 0, "average": 0}
        return {
            "count": len(self.data),
            "sum": sum(self.data),
            "average": sum(self.data) / len(self.data)
        }
'''
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(example_code)
with open(file_path, "r", encoding="utf-8") as f:
    code_content = f.read()

# ============ 调用模型分析 ============
result = chain.invoke({"code": code_content})

print("\n AI 代码审查结果：\n")
print(result.content)
