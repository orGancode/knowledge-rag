from typing import List
from langchain.schema import Document
import pypdf


class PDFLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
    
    def load(self) -> List[Document]:
        # 1. 使用pypdf打开PDF
        documents = []
        
        with open(self.file_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            
            # 2. 逐页提取文本，保留页码作为metadata
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                
                # 关键：metadata必须包含source和page
                # 这对后续溯源至关重要
                metadata = {
                    "source": self.file_path,
                    "page": page_num + 1  # 页码从1开始
                }
                
                documents.append(Document(page_content=text, metadata=metadata))
        
        # 3. 返回Document对象列表
        return documents