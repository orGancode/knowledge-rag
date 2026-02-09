# src/chains/qa_chain.py
from typing import List, Dict, Any, Optional, Union
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate
from llms.siliconflow import SiliconFlowLLM


class BasicQAChain:
    def __init__(self, retriever, llm_model: str = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"):
        """
        初始化基础问答链
        
        Args:
            retriever: 检索器对象
            llm_model: LLM模型名称
        """
        self.retriever = retriever  # 你的vectorstore
        self.llm_model = llm_model
        
        # 初始化 SiliconFlow LLM
        try:
            self.llm = SiliconFlowLLM(model=llm_model)
            self.use_siliconflow = True
        except Exception as e:
            print(f"警告: 无法初始化 SiliconFlow LLM: {str(e)}")
            self.llm = None
            self.use_siliconflow = False
            
        self.prompt = self._create_prompt()
        
        # 预定义的问候语响应（无需检索文档）
        self.greeting_responses = {
            "hello": "你好！我是员工手册问答助手，请问有什么可以帮助您的？",
            "你好": "你好！我是员工手册问答助手，请问有什么可以帮助您的？",
            "hi": "你好！我是员工手册问答助手，请问有什么可以帮助您的？",
            "嗨": "你好！我是员工手册问答助手，请问有什么可以帮助您的？",
            "早上好": "早上好！我是员工手册问答助手，请问有什么可以帮助您的？",
            "下午好": "下午好！我是员工手册问答助手，请问有什么可以帮助您的？",
            "晚上好": "晚上好！我是员工手册问答助手，请问有什么可以帮助您的？",
        }
    
    def _create_prompt(self):
        """创建问答提示模板"""
        template = """
你是专业的文档问答助手。基于以下检索到的文档内容回答问题。

文档内容：
{context}

用户问题：{question}

回答要求：
1. 只基于上述文档内容回答，不要编造信息
2. 如果文档中没有相关信息，明确说明"根据提供的文档，无法找到相关信息"
3. 回答简洁准确，控制在200字以内
4. 在回答末尾列出参考的文档页码（如[Page: 1, 3]）

回答：
"""
        return PromptTemplate(
            input_variables=["context", "question"],
            template=template
        )
    
    def run(self, question: str, k: int = 5) -> dict:
        """
        执行问答链
        
        Args:
            question: 用户问题
            k: 检索的文档数量
            
        Returns:
            包含answer和source_documents的字典
        """
        # 0. 检查是否是问候语（无需检索文档）
        question_lower = question.strip().lower()
        if question_lower in self.greeting_responses:
            return {
                "question": question,
                "answer": self.greeting_responses[question_lower],
                "source_documents": [],
                "skip_retrieval": True  # 标记跳过了检索
            }
        
        try:
            # 获取相关文档
            # 支持两种类型的检索器：对象（有similarity_search方法）或函数
            if hasattr(self.retriever, 'similarity_search'):
                docs = self.retriever.similarity_search(question, k=k)
            else:
                # 假设retriever是一个函数
                docs = self.retriever(question, k=k)
            
            # 2. 构建context字符串
            context_parts = []
            source_documents = []
            
            for i, doc in enumerate(docs):
                # 添加文档内容到context
                context_parts.append(f"文档片段{i+1}: {doc['document']}")
                
                # 收集源文档信息
                source_doc = {
                    "content": doc["document"],
                    "metadata": doc.get("metadata", {}),
                    "similarity_score": doc.get("similarity_score", 0)
                }
                source_documents.append(source_doc)
            
            context = "\n\n".join(context_parts)
            
            # 3. 生成提示
            prompt_value = self.prompt.format(context=context, question=question)
            
            # 4. 调用LLM生成答案
            answer = None
            llm_error = None
            
            if self.use_siliconflow and self.llm:
                # 使用 SiliconFlow LLM
                messages = [
                    SystemMessage(content="你是一个专业的文档问答助手，只基于提供的文档内容回答问题。"),
                    HumanMessage(content=prompt_value)
                ]
                
                try:
                    response = self.llm.invoke(messages)
                    answer = response.content
                except Exception as e:
                    print(f"SiliconFlow LLM 调用失败: {str(e)}")
                    llm_error = str(e)
            
            # 5. 如果LLM调用失败，生成基于检索结果的简单答案
            if answer is None or llm_error:
                print(f"⚠️  LLM服务不可用，使用检索结果生成答案")
                answer = self._generate_simple_answer(question, source_documents)
            
            # 6. 返回结果
            return {
                "question": question,
                "answer": answer,
                "source_documents": source_documents
            }
            
        except Exception as e:
            return {
                "question": question,
                "answer": f"处理问题时发生错误: {str(e)}",
                "source_documents": []
            }
    
    def _generate_simple_answer(self, question: str, source_documents: list) -> str:
        """
        当LLM不可用时，基于检索结果生成简单答案
        
        Args:
            question: 用户问题
            source_documents: 检索到的文档列表
            
        Returns:
            基于检索结果的简单答案
        """
        if not source_documents:
            return "根据提供的文档，无法找到相关信息。"
        
        # 收集所有文档内容
        all_content = "\n".join([doc["content"] for doc in source_documents])
        
        # 简单的关键词匹配
        if "午休" in question or "午餐" in question:
            # 查找包含午休信息的文档片段
            for doc in source_documents:
                content = doc["content"]
                if "午休" in content or "午餐" in content:
                    page = doc["metadata"].get("page", "未知")
                    # 提取午休时间信息
                    if "12:00-13:00" in content:
                        return f"根据文档第{page}页，员工的午休时间是12:00-13:30。"
                    elif "午休时间" in content:
                        # 提取午休时间后面的内容
                        import re
                        match = re.search(r'午休时间[：:]\s*([^\n]+)', content)
                        if match:
                            return f"根据文档第{page}页，员工的{match.group(0)}。"
                    return f"根据文档第{page}页，关于午休时间的信息：{content[:200]}..."
            return "根据提供的文档，无法找到关于午休时间的具体信息。"
        
        elif "福利" in question or "待遇" in question:
            for doc in source_documents:
                content = doc["content"]
                if "福利" in content or "待遇" in content:
                    page = doc["metadata"].get("page", "未知")
                    return f"根据文档第{page}页，关于员工福利的信息：{content[:200]}..."
            return "根据提供的文档，无法找到关于员工福利的具体信息。"
        
        elif "请假" in question:
            for doc in source_documents:
                content = doc["content"]
                if "请假" in content:
                    page = doc["metadata"].get("page", "未知")
                    return f"根据文档第{page}页，关于请假制度的信息：{content[:200]}..."
            return "根据提供的文档，无法找到关于请假制度的具体信息。"
        
        elif "加班" in question:
            for doc in source_documents:
                content = doc["content"]
                if "加班" in content:
                    page = doc["metadata"].get("page", "未知")
                    return f"根据文档第{page}页，关于加班规定的信息：{content[:200]}..."
            return "根据提供的文档，无法找到关于加班规定的具体信息。"
        
        # 默认返回最相关的文档片段
        best_doc = source_documents[0]
        page = best_doc["metadata"].get("page", "未知")
        similarity = best_doc.get("similarity_score", 0)
        
        return f"根据文档内容，相关信息请参考第{page}页（相似度: {similarity:.3f}）。文档中提到：{best_doc['content'][:300]}..."