#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
决策管理 Agent (Decision Manager)
--------------------------------
功能：
- 任务调度与协调：管理项目分析、漏洞检测、漏洞修复三个Agent的工作流程
- 状态跟踪：监控各Agent执行状态和结果
- 决策制定：根据分析结果决定下一步行动策略
- 报告生成：整合各Agent结果，生成综合报告

特点：
- 支持串行和并行执行模式
- 错误处理和重试机制
- 可配置的执行策略
- 详细的执行日志和状态报告
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("DecisionManager")


class AgentStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class AgentResult:
    agent_name: str
    status: AgentStatus
    output_path: Optional[Path] = None
    error_message: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskContext:
    project_path: Path
    output_dir: Path
    config: Dict[str, Any]
    results: Dict[str, AgentResult] = field(default_factory=dict)


class DecisionManager:
    def __init__(
            self,
            project_path: str,
            output_dir: str = "ai_agent_output",
            config_file: Optional[str] = None,
            max_workers: int = 2
    ):
        self.project_path = Path(project_path).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 加载配置
        self.config = self._load_config(config_file)
        self.max_workers = max_workers

        # Agent 执行配置
        self.agents_config = {
            "project_analyzer": {
                "script": str(Path(__file__).parent / "ai_subagent.py"),  # 使用绝对路径
                "args": [
                    "--root", str(self.project_path),
                    "--output", str(self.output_dir / "project_analysis.json"),
                    "--format", "json",
                    "--lan", "zh"
                ],
                "dependencies": [],
                "timeout": 600
            },
            "vulnerability_detector": {
                "script": str(Path(__file__).parent / "main.py"),
                "args": [],
                "dependencies": ["project_analyzer"],
                "timeout": 300
            },
            "vulnerability_repair": {
                "script": str(Path(__file__).parent / "ai_repair_agent(1).py"),
                "args": [
                    "run",
                    "--path", str(self.project_path),
                    "--ext", ".py",
                    "--dry-run",
                    "--save", str(self.output_dir / "repair_report.json")
                ],
                "dependencies": ["vulnerability_detector"],
                "timeout": 600
            }
        }

        self.context = TaskContext(
            project_path=self.project_path,
            output_dir=self.output_dir,
            config=self.config
        )

    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """加载配置文件"""
        default_config = {
            "execution_mode": "sequential",  # sequential 或 parallel
            "enable_repair": False,  # 是否启用自动修复
            "backup_files": True,
            "max_retries": 2,
            "timeout_multiplier": 1.5
        }

        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                logger.warning(f"配置文件加载失败: {e}，使用默认配置")

        return default_config

    def _run_agent(self, agent_name: str, agent_config: Dict[str, Any]) -> AgentResult:
        """执行单个Agent"""
        result = AgentResult(agent_name=agent_name, status=AgentStatus.RUNNING)
        start_time = time.time()

        try:
            script_path = Path(agent_config["script"])
            if not script_path.exists():
                raise FileNotFoundError(f"Agent脚本不存在: {script_path}")

            # 构建命令
            cmd = [sys.executable, str(script_path)] + agent_config["args"]

            logger.info(f"执行Agent: {agent_name}")
            logger.debug(f"命令: {' '.join(cmd)}")

            # 执行命令
            timeout = agent_config.get("timeout", 300)
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=Path.cwd()
            )

            result.execution_time = time.time() - start_time

            if process.returncode == 0:
                result.status = AgentStatus.COMPLETED
                # 设置输出路径
                if agent_name == "project_analyzer":
                    result.output_path = self.output_dir / "project_analysis.json"
                elif agent_name == "vulnerability_repair":
                    result.output_path = self.output_dir / "repair_report.json"
                logger.info(f"Agent {agent_name} 执行成功，耗时: {result.execution_time:.2f}s")
            else:
                result.status = AgentStatus.FAILED
                result.error_message = f"退出码: {process.returncode}, 错误: {process.stderr}"
                logger.error(f"Agent {agent_name} 执行失败: {result.error_message}")

        except subprocess.TimeoutExpired:
            result.status = AgentStatus.FAILED
            result.error_message = f"执行超时 (>{agent_config['timeout']}s)"
            logger.error(f"Agent {agent_name} 执行超时")
        except Exception as e:
            result.status = AgentStatus.FAILED
            result.error_message = str(e)
            logger.error(f"Agent {agent_name} 执行异常: {e}")

        return result

    def _check_dependencies(self, agent_name: str) -> bool:
        """检查Agent依赖是否满足"""
        dependencies = self.agents_config[agent_name].get("dependencies", [])
        for dep in dependencies:
            dep_result = self.context.results.get(dep)
            if not dep_result or dep_result.status != AgentStatus.COMPLETED:
                logger.warning(f"Agent {agent_name} 依赖 {dep} 未完成，跳过执行")
                return False
        return True

    def execute_sequential(self) -> bool:
        """顺序执行所有Agent"""
        logger.info("开始顺序执行Agent工作流")

        agent_execution_order = ["project_analyzer", "vulnerability_detector", "vulnerability_repair"]

        for agent_name in agent_execution_order:
            if not self._check_dependencies(agent_name):
                self.context.results[agent_name] = AgentResult(
                    agent_name=agent_name,
                    status=AgentStatus.SKIPPED,
                    error_message="依赖未满足"
                )
                continue

            result = self._run_agent(agent_name, self.agents_config[agent_name])
            self.context.results[agent_name] = result

            if result.status == AgentStatus.FAILED:
                logger.error(f"Agent {agent_name} 执行失败，停止工作流")
                return False

        return True

    def execute_parallel(self) -> bool:
        """并行执行可并行的Agent"""
        logger.info("开始并行执行Agent工作流")

        # 识别可以并行执行的Agent
        executable_agents = []
        for agent_name, config in self.agents_config.items():
            if self._check_dependencies(agent_name):
                executable_agents.append((agent_name, config))

        # 使用线程池并行执行
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_agent = {
                executor.submit(self._run_agent, agent_name, config): agent_name
                for agent_name, config in executable_agents
            }

            for future in as_completed(future_to_agent):
                agent_name = future_to_agent[future]
                try:
                    result = future.result()
                    self.context.results[agent_name] = result
                except Exception as e:
                    logger.error(f"Agent {agent_name} 执行异常: {e}")
                    self.context.results[agent_name] = AgentResult(
                        agent_name=agent_name,
                        status=AgentStatus.FAILED,
                        error_message=str(e)
                    )

        # 检查是否有失败的Agent
        return all(
            result.status != AgentStatus.FAILED
            for result in self.context.results.values()
        )

    def generate_final_report(self) -> Dict[str, Any]:
        """生成最终报告"""
        report = {
            "project_info": {
                "project_path": str(self.project_path),
                "analysis_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "output_directory": str(self.output_dir)
            },
            "execution_summary": {
                "total_agents": len(self.context.results),
                "completed": sum(1 for r in self.context.results.values() if r.status == AgentStatus.COMPLETED),
                "failed": sum(1 for r in self.context.results.values() if r.status == AgentStatus.FAILED),
                "skipped": sum(1 for r in self.context.results.values() if r.status == AgentStatus.SKIPPED)
            },
            "agent_results": {},
            "recommendations": []
        }

        # 收集各Agent结果
        for agent_name, result in self.context.results.items():
            report["agent_results"][agent_name] = {
                "status": result.status.value,
                "execution_time": result.execution_time,
                "error_message": result.error_message,
                "output_path": str(result.output_path) if result.output_path else None
            }

            # 尝试加载Agent输出内容
            if result.output_path and result.output_path.exists():
                try:
                    with open(result.output_path, 'r', encoding='utf-8') as f:
                        agent_output = json.load(f)
                    report["agent_results"][agent_name]["output_summary"] = self._summarize_agent_output(agent_name,
                                                                                                         agent_output)
                except Exception as e:
                    logger.warning(f"无法加载Agent {agent_name} 的输出: {e}")

        # 生成建议
        report["recommendations"] = self._generate_recommendations()

        return report

    def _summarize_agent_output(self, agent_name: str, output: Dict[str, Any]) -> Dict[str, Any]:
        """汇总Agent输出"""
        if agent_name == "project_analyzer":
            return {
                "project_overview": output.get("project_overview", "N/A"),
                "file_count": len(output.get("files", [])),
                "architecture_style": output.get("architecture", {}).get("style", "unknown")
            }
        elif agent_name == "vulnerability_repair":
            patches = output.get("补丁", [])
            return {
                "issues_found": len(output.get("问题列表", [])),
                "patches_generated": len(patches),
                "patch_types": [p.get("补丁类型", "unknown") for p in patches]
            }
        else:
            return {"raw_output_available": True}

    def _generate_recommendations(self) -> List[str]:
        """基于分析结果生成建议"""
        recommendations = []

        # 检查项目分析结果
        project_result = self.context.results.get("project_analyzer")
        if project_result and project_result.status == AgentStatus.COMPLETED:
            recommendations.append("✅ 项目结构分析完成，建议查看详细架构文档")

        # 检查漏洞检测结果
        vuln_result = self.context.results.get("vulnerability_detector")
        if vuln_result and vuln_result.status == AgentStatus.COMPLETED:
            recommendations.append("🔍 代码审查完成，建议修复发现的潜在问题")

        # 检查修复结果
        repair_result = self.context.results.get("vulnerability_repair")
        if repair_result:
            if repair_result.status == AgentStatus.COMPLETED:
                recommendations.append("🔧 修复建议已生成，请审阅后应用补丁")
            elif repair_result.status == AgentStatus.FAILED:
                recommendations.append("⚠️ 修复Agent执行失败，建议手动检查代码问题")

        # 总体建议
        if all(r.status == AgentStatus.COMPLETED for r in self.context.results.values()):
            recommendations.append(" 所有分析完成，建议进行人工验证后部署")
        else:
            failed_agents = [name for name, r in self.context.results.items() if r.status == AgentStatus.FAILED]
            recommendations.append(f"❌ 以下Agent执行失败: {', '.join(failed_agents)}，建议检查配置后重试")

        return recommendations

    def run(self) -> bool:
        """执行完整的工作流"""
        logger.info(f"开始分析项目: {self.project_path}")

        # 选择执行模式
        execution_mode = self.config.get("execution_mode", "sequential")
        success = False

        if execution_mode == "parallel":
            success = self.execute_parallel()
        else:
            success = self.execute_sequential()

        # 生成报告
        final_report = self.generate_final_report()
        report_path = self.output_dir / "final_report.json"

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, ensure_ascii=False, indent=2)

        logger.info(f"分析完成，最终报告保存至: {report_path}")

        # 打印简要结果
        self._print_summary(final_report)

        return success

    def _print_summary(self, report: Dict[str, Any]):
        """打印执行摘要"""
        print("\n" + "=" * 50)
        print("AI Agent 协作分析结果摘要")
        print("=" * 50)

        summary = report["execution_summary"]
        print(f"项目路径: {report['project_info']['project_path']}")
        print(f"分析时间: {report['project_info']['analysis_date']}")
        print(f"Agent完成情况: {summary['completed']}/{summary['total_agents']}")
        print(f"失败: {summary['failed']}, 跳过: {summary['skipped']}")

        print("\n详细结果:")
        for agent_name, result in report["agent_results"].items():
            status_icon = "✅" if result["status"] == "completed" else "❌" if result["status"] == "failed" else "⚠️"
            print(f"  {status_icon} {agent_name}: {result['status']} ({result['execution_time']:.2f}s)")

        print("\n建议:")
        for rec in report["recommendations"]:
            print(f"  • {rec}")

        print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="AI Agent 决策管理器 - 协调项目分析、漏洞检测和修复")
    parser.add_argument(
        "--project",
        type=str,
        required=True,
        help="要分析的项目路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ai_agent_output",
        help="输出目录 (默认: ai_agent_output)"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="配置文件路径 (JSON格式)"
    )
    parser.add_argument(
        "--mode",
        choices=["sequential", "parallel"],
        default="sequential",
        help="执行模式 (默认: sequential)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="并行工作线程数 (默认: 2)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="启用详细日志"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # 创建决策管理器并执行
    manager = DecisionManager(
        project_path=args.project,
        output_dir=args.output,
        config_file=args.config,
        max_workers=args.workers
    )

    # 更新配置
    if args.mode:
        manager.config["execution_mode"] = args.mode

    success = manager.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()