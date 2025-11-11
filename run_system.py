#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Agent 协作系统启动脚本
"""

import os
import sys
from pathlib import Path

# 添加agent目录到Python路径
agents_dir = Path(__file__).parent / "agents"
sys.path.insert(0, str(agents_dir))

from decision_manager import DecisionManager


def main():
    """启动AI Agent协作系统"""

    # 配置参数
    project_path = input("请输入要分析的项目路径: ").strip()
    if not project_path:
        project_path = "."  # 默认当前目录

    output_dir = "ai_agent_outputs"
    config_file = "config/agent_config.json"

    # 创建决策管理器
    manager = DecisionManager(
        project_path=project_path,
        output_dir=output_dir,
        config_file=config_file if Path(config_file).exists() else None,
        max_workers=2
    )

    print(" 启动AI Agent协作系统...")
    print(f" 项目路径: {project_path}")
    print(f" 输出目录: {output_dir}")
    print("=" * 50)

    # 执行分析
    success = manager.run()

    if success:
        print("\n 分析完成！")
    else:
        print("\n⚠️ 分析过程中遇到问题，请检查日志")

    return success


if __name__ == "__main__":
    main()