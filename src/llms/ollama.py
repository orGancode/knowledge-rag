# src/llms/ollama_llm.py

import requests
from typing import Iterator, Union, List, Any

class OllamaLLM:
    def __init__(self, model: str = "qwen:4b", temperature: float = 0.7, timeout: int = 120):
        self.model = model
        self.temperature = temperature
        self.base_url = "http://localhost:11434"
        self.timeout = timeout  # 增加到 120 秒
    
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
            # 将消息列表转换为单个提示字符串
            prompt_text = self._messages_to_prompt(prompt)
            
            # 调用 Ollama API
            response_text = self._call_ollama_api(prompt_text)
            
            # 返回一个包含 content 属性的对象（模拟 LangChain 的响应格式）
            class Response:
                def __init__(self, content: str):
                    self.content = content
            
            return Response(response_text)
        else:
            # 字符串格式，直接调用
            return self._call_ollama_api(str(prompt))
    
    def _messages_to_prompt(self, messages: List[Any]) -> str:
        """
        将 LangChain 消息列表转换为单个提示字符串
        
        Args:
            messages: 消息列表（如 [SystemMessage, HumanMessage]）
            
        Returns:
            格式化的提示字符串
        """
        prompt_parts = []
        
        for msg in messages:
            # 获取消息内容
            if hasattr(msg, 'content'):
                content = msg.content
            else:
                content = str(msg)
            
            # 根据消息类型添加前缀
            msg_type = type(msg).__name__
            if msg_type == 'SystemMessage':
                prompt_parts.append(f"系统指令: {content}")
            elif msg_type == 'HumanMessage':
                prompt_parts.append(f"用户问题: {content}")
            elif msg_type == 'AIMessage':
                prompt_parts.append(f"助手回答: {content}")
            else:
                prompt_parts.append(content)
        
        return "\n\n".join(prompt_parts)
    
    def _call_ollama_api(self, prompt: str) -> str:
        """
        调用 Ollama API
        
        Args:
            prompt: 提示字符串
            
        Returns:
            模型响应
        """
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "temperature": self.temperature,
                "stream": False  # 非流式，等待完整响应
            },
            timeout=self.timeout  # 关键：延长超时
        )
        response.raise_for_status()
        return response.json()["response"]
    
    def stream(self, prompt: str) -> Iterator[str]:
        """流式生成，减少等待焦虑"""
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "temperature": self.temperature,
                "stream": True
            },
            timeout=self.timeout,
            stream=True
        )
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                if "response" in data:
                    yield data["response"]
                if data.get("done"):
                    break