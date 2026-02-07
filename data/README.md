# 添加你的测试 PDF 文件到这里

这个目录用于存放 RAG 系统需要处理的文档。支持的格式：
- PDF (.pdf)
- 文本文件 (.txt)
- Word 文档 (.docx)

使用方式：
1. 将你的文档复制到本目录
2. 在代码中引用：loaders.PDFLoader().load('data/sample.pdf')
