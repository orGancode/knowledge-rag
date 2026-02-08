#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试基于标题的PDF分割器
"""

import sys
import os

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from loaders.pdf_loader import PDFLoader
from splitters import HeadingBasedSplitter

def test_heading_splitter():
    """测试基于标题的分割器"""
    
    print("=" * 80)
    print("测试基于标题的PDF分割器")
    print("=" * 80)
    
    # 1. 加载PDF文档
    print("\n📄 加载PDF文档...")
    loader = PDFLoader("data/员工手册.pdf")
    documents = loader.load()
    print(f"   共加载 {len(documents)} 页文档")
    
    # 2. 使用基于标题的分割器
    print("\n✂️  使用基于标题的分割器...")
    splitter = HeadingBasedSplitter(
        max_chunk_size=1500,
        chunk_overlap=100
    )
    chunks = splitter.split(documents)
    print(f"   共生成 {len(chunks)} 个文本块")
    
    # 3. 显示每个文本块的信息
    print("\n📋 文本块详情:")
    print("-" * 80)
    
    for i, chunk in enumerate(chunks, 1):
        content = chunk.page_content
        metadata = chunk.metadata
        
        print(f"\n[文本块 {i}]")
        print(f"  页码: {metadata.get('page', '未知')}")
        print(f"  长度: {len(content)} 字符")
        print(f"  分割器类型: {metadata.get('splitter_type', '未知')}")
        print(f"  检测到的标题数: {metadata.get('headings_count', 0)}")
        
        # 显示前200个字符
        preview = content[:200].replace('\n', ' ')
        print(f"  内容预览: {preview}...")
        
        # 检测并显示标题
        lines = content.split('\n')
        print(f"  标题行:")
        for line in lines[:5]:  # 只显示前5行
            stripped = line.strip()
            if stripped and len(stripped) < 100:  # 可能是标题
                print(f"    - {stripped}")
    
    # 4. 统计信息
    print("\n" + "=" * 80)
    print("📊 统计信息:")
    print("-" * 80)
    
    total_chars = sum(len(chunk.page_content) for chunk in chunks)
    avg_chars = total_chars / len(chunks) if chunks else 0
    
    print(f"  文本块总数: {len(chunks)}")
    print(f"  总字符数: {total_chars}")
    print(f"  平均每块字符数: {avg_chars:.1f}")
    print(f"  最小块字符数: {min(len(chunk.page_content) for chunk in chunks) if chunks else 0}")
    print(f"  最大块字符数: {max(len(chunk.page_content) for chunk in chunks) if chunks else 0}")
    
    # 5. 对比：使用递归分割器
    print("\n" + "=" * 80)
    print("🔄 对比：使用递归分割器")
    print("-" * 80)
    
    from splitters import RecursiveTextSplitter
    recursive_splitter = RecursiveTextSplitter(chunk_size=1000, chunk_overlap=200)
    recursive_chunks = recursive_splitter.split(documents)
    
    print(f"  递归分割器生成: {len(recursive_chunks)} 个文本块")
    print(f"  基于标题分割器生成: {len(chunks)} 个文本块")
    print(f"  文本块数量变化: {len(chunks) - len(recursive_chunks):+d} ({((len(chunks)/len(recursive_chunks)-1)*100):+.1f}%)")
    
    print("\n" + "=" * 80)
    print("✅ 测试完成！")
    print("=" * 80)

if __name__ == "__main__":
    test_heading_splitter()
