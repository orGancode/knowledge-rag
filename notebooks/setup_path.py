# notebooks/setup_path.py
"""
在Notebook开头导入此模块即可使用src中的代码
用法: import setup_path
"""
import sys
from pathlib import Path

def setup_project_path():
    """将项目根目录加入Python路径"""
    # 当前文件路径 → notebooks/ → 项目根目录
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent  # 上两级到rag-demo/
    
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        print(f"✅ 已添加项目路径: {project_root}")
    else:
        print(f"路径已存在: {project_root}")
    
    return project_root

# 自动执行
PROJECT_ROOT = setup_project_path()