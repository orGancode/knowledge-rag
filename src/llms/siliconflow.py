# src/llms/siliconflow.py

import requests
from typing import Iterator, Union, List, Any
import os


class SiliconFlowLLM:
    """SiliconFlow API LLM客户端"""
    
    def __init__(
        self,
        model: str = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        api_key: str = None,
        base_url: str = "https://api.siliconflow.cn/v1",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        timeout: int = 120
    ):
        """
        初始化SiliconFlow LLM客户端
        
        Args:
            model: 模型名称
            api_key: SiliconFlow API密钥
            base_url: API基础URL
            temperature: 温度参数
            max_tokens: 最大生成token数
            timeout: 请求超时时间（秒）
        """
        self.model = model
        self.api_key = api_key or os.getenv("SILICONFLOW_API_KEY")
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        
        if not self.api_key:
            raise ValueError("SiliconFlow API密钥未提供，请设置SILICONFLOW_API_KEY环境变量或传入api_key参数")
    
    def invoke(self, prompt: Union[str, List[Any]]) -> Union[str, Any]:
        """
        同步调用，支持字符串或消息列表格式
        
        Args:
            prompt: 可以是字符串或消息列表（如 [SystemMessage, HumanMessage]）
            
        Returns:
            如果输入是字符串，返回字符串响应
            如果输入是消息列表，返回包含 content 属性的对象
        """
        # 检查是否是消息列表格式
        if isinstance(prompt, list) and len(prompt) > 0:
            # 将消息列表转换为API格式
            messages = self._messages_to_api_format(prompt)
            
            # 调用 SiliconFlow API
            response_text = self._call_siliconflow_api(messages)
            
            # 返回一个包含 content 属性的对象（模拟 LangChain 的响应格式）
            class Response:
                def __init__(self, content: str):
                    self.content = content
            
            return Response(response_text)
        else:
            # 字符串格式，转换为消息格式
            messages = [{"role": "user", "content": str(prompt)}]
            return self._call_siliconflow_api(messages)
    
    def _messages_to_api_format(self, messages: List[Any]) -> List[dict]:
        """
        将 LangChain 消息列表转换为 SiliconFlow API 格式
        
        Args:
            messages: 消息列表（如 [SystemMessage, HumanMessage]）
            
        Returns:
            API格式的消息列表
        """
        api_messages = []
        
        for msg in messages:
            # 获取消息内容
            if hasattr(msg, 'content'):
                content = msg.content
            else:
                content = str(msg)
            
            # 根据消息类型映射到API角色
            msg_type = type(msg).__name__
            if msg_type == 'SystemMessage':
                api_messages.append({"role": "system", "content": content})
            elif msg_type == 'HumanMessage':
                api_messages.append({"role": "user", "content": content})
            elif msg_type == 'AIMessage':
                api_messages.append({"role": "assistant", "content": content})
            else:
                # 默认作为用户消息
                api_messages.append({"role": "user", "content": content})
        
        return api_messages
    
    def _call_siliconflow_api(self, messages: List[dict]) -> str:
        """
        调用 SiliconFlow API
        
        Args:
            messages: 消息列表
            
        Returns:
            模型响应
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False
        }
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout
        )
        
        response.raise_for_status()
        
        # 解析响应
        result = response.json()
        return result["choices"][0]["message"]["content"]
    
    def stream(self, prompt: Union[str, List[Any]]) -> Iterator[str]:
        """
        流式生成
        
        Args:
            prompt: 可以是字符串或消息列表
            
        Yields:
            生成的文本片段
        """
        # 转换为消息格式
        if isinstance(prompt, list) and len(prompt) > 0:
            messages = self._messages_to_api_format(prompt)
        else:
            messages = [{"role": "user", "content": str(prompt)}]
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True
        }
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout,
            stream=True
        )
        
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data_str = line[6:]  # 移除 'data: ' 前缀
                    if data_str == '[DONE]':
                        break
                    try:
                        import json
                        data = json.loads(data_str)
                        if "choices" in data and len(data["choices"]) > 0:
                            delta = data["choices"][0].get("delta", {})
                            if "content" in delta:
                                yield delta["content"]
                    except json.JSONDecodeError:
                        continue
