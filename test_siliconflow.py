# 测试 SiliconFlow LLM

import os
from dotenv import load_dotenv
from src.llms.siliconflow import SiliconFlowLLM

# 加载环境变量
load_dotenv()

# 测试 SiliconFlow LLM
print("=" * 60)
print("测试 SiliconFlow LLM")
print("=" * 60)

try:
    # 初始化 LLM
    llm = SiliconFlowLLM(
        model="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        temperature=0.7
    )
    
    print("\n✅ SiliconFlow LLM 初始化成功")
    print(f"模型: {llm.model}")
    print(f"API Base URL: {llm.base_url}")
    
    # 测试简单调用
    print("\n📝 测试简单调用...")
    response = llm.invoke("你好，请用一句话介绍你自己。")
    print(f"响应: {response}")
    
    # 测试消息列表格式
    print("\n📝 测试消息列表格式...")
    from langchain.schema import SystemMessage, HumanMessage
    
    messages = [
        SystemMessage(content="你是一个专业的文档问答助手。"),
        HumanMessage(content="什么是RAG？")
    ]
    
    response = llm.invoke(messages)
    print(f"响应: {response.content}")
    
    print("\n✅ 所有测试通过！")
    
except Exception as e:
    print(f"\n❌ 测试失败: {str(e)}")
    print("\n💡 请确保：")
    print("1. 已在 .env 文件中设置 SILICONFLOW_API_KEY")
    print("2. API Key 有效且有足够的额度")
    print("3. 网络连接正常")

print("=" * 60)
