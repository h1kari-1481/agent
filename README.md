# 云南省企业就业失业数据采集系统
## 基于 Agent 编程的软件系统

> 软工大作业 3 – Agent-based Software System

---

## 项目简介

本系统模拟「**云南省企业就业失业数据采集系统**」，采用多智能体（Multi-Agent）架构，将系统拆分为四个职责独立的 Agent，通过消息总线协作，完成企业就业数据的采集、审核、汇总与分析全流程。

---

## 架构概览

```
┌─────────────────────────────────────────────────────────┐
│                      Flask REST API                      │
│                     (src/api/app.py)                     │
└────────────────────────┬────────────────────────────────┘
                         │ Message
                    ┌────▼────┐
                    │AgentBus │  消息路由
                    └────┬────┘
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼               ▼
  ┌──────────┐   ┌──────────────┐  ┌──────────┐  ┌──────────────┐
  │  System  │   │  Enterprise  │  │   City   │  │  Province    │
  │  Agent   │   │    Agent     │  │  Agent   │  │   Agent      │
  └──────────┘   └──────────────┘  └──────────┘  └──────────────┘
       │                │                │                │
       └────────────────┴────────────────┴────────────────┘
                                    │
                           ┌────────▼────────┐
                           │  SQLite Database │
                           │  (data/system.db)│
                           └─────────────────┘
```

| Agent | 职责 |
|---|---|
| **SystemAgent** | 用户认证、用户/角色管理、系统监控、消息审计 |
| **EnterpriseAgent** | 企业备案、数据填报、历史查询、通知浏览 |
| **CityAgent** | 待审列表、审核通过/退回、企业查询 |
| **ProvinceAgent** | 备案审核、数据汇总、多维分析、通知发布、上报部委 |

---

## 快速启动

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动服务

```bash
python main.py
```

服务将在 `http://localhost:5000` 启动。

### 3. API 使用示例

#### 登录

```bash
curl -X POST http://localhost:5000/api/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin123"}'
```

返回 `token`，后续请求带入 `X-Auth-Token: <token>` 头。

#### 创建调查期

```bash
curl -X POST http://localhost:5000/api/province/create_survey_period \
  -H "X-Auth-Token: <token>" \
  -H "Content-Type: application/json" \
  -d '{"period_key":"2026-01","start_date":"2026-01-01","end_date":"2026-01-31"}'
```

#### 企业注册

```bash
curl -X POST http://localhost:5000/api/enterprise/register_enterprise \
  -H "X-Auth-Token: <token>" \
  -H "Content-Type: application/json" \
  -d '{"org_code":"ENT001","name":"测试科技有限公司","nature":"私营",
       "industry":"信息技术","region_code":"5301",
       "contact_person":"张三","contact_phone":"0871-12345678",
       "email":"test@example.com","address":"昆明市盘龙区"}'
```

#### 数据填报

```bash
curl -X POST http://localhost:5000/api/enterprise/submit_data \
  -H "X-Auth-Token: <token>" \
  -H "Content-Type: application/json" \
  -d '{"enterprise_id":1,"period_id":1,
       "baseline_employees":100,"survey_employees":90,
       "decrease_type":"经济性裁员",
       "main_reason":"订单不足","main_reason_desc":"受市场影响"}'
```

#### 省局汇总报告

```bash
curl -X POST http://localhost:5000/api/province/aggregate_report \
  -H "X-Auth-Token: <token>" \
  -H "Content-Type: application/json" \
  -d '{"period_key":"2026-01"}'
```

---

## 目录结构

```
agent/
├── main.py                  # 服务入口
├── generate_docs.py         # 文档生成脚本
├── requirements.txt
├── data/                    # SQLite 数据库（运行时生成）
├── docs/                    # 交付文档
│   ├── 项目计划.docx
│   ├── 变更单_CR-001.docx
│   ├── 变更单_CR-002.docx
│   ├── 变更单_CR-003.docx
│   └── 迭代历史说明.docx
├── src/
│   ├── agents/
│   │   ├── base_agent.py     # 抽象基类 & Message
│   │   ├── agent_bus.py      # 消息总线
│   │   ├── enterprise_agent.py
│   │   ├── city_agent.py
│   │   ├── province_agent.py
│   │   └── system_agent.py
│   ├── models/
│   │   └── database.py       # SQLite 初始化
│   └── api/
│       └── app.py            # Flask REST API
└── tests/
    └── test_agents.py        # 17 个单元/集成测试
```

---

## 运行测试

```bash
python -m pytest tests/ -v
```

预期输出：**17 passed**

---

## 数据流程

```
企业用户                市局                省局
   │                    │                   │
   │─register_enterprise─▶                  │
   │                    │                   │
   │                    │  ◀─approve_enterprise─│
   │                    │                   │
   │─submit_data────────▶                   │
   │                    │─approve_submission─▶  │
   │                    │                   │─aggregate_report
   │                    │                   │─forward_to_ministry
```

---

## 提交清单

| 序号 | 文件 | 说明 |
|---|---|---|
| 1 | `docs/项目计划.docx` | 项目计划（含甘特图） |
| 2 | `docs/迭代历史说明.docx` | CM 工具迭代历史截图说明 |
| 3 | `docs/变更单_CR-001.docx` | 变更单1：多维分析扩展 |
| 4 | `docs/变更单_CR-002.docx` | 变更单2：密码安全加强 |
| 5 | `docs/变更单_CR-003.docx` | 变更单3：Excel导出功能 |
| 6 | `src/` | 基于 Agent 编程的软件系统源码 |
| 7 | `tests/` | 自动化测试（17个用例全部通过） |
