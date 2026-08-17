# 医疗 AI Agent 学习项目

这是一个基于 LangGraph、Qwen Structured Output、受控 Tool Calling 和 SQLite 的医疗
Agent 学习项目。当前业务数据全部是骨质疏松访视表生成的合成数据，不能用于诊疗。

## 当前运行架构

项目不再提供离线 Demo 模式。网页和 CLI 都使用 Qwen，入口也不再让用户选择运行模式。

```text
用户最新消息 + 最近对话 + 当前患者 + 上次查询范围
                         ↓
              Qwen QueryPlan Structured Output
                         ↓
          Python Schema 与54项字段白名单校验
                         ↓
     patient_data / capability / general_chat / unclear
                         ↓
         Patient ID → Authorization → Tool → 安全渲染
```

原先独立的规则 Router 和指标选择节点已经从运行图移除。Qwen现在一次输出统一计划：

```json
{
  "intent": "patient_data",
  "patient_reference": "current",
  "scope": "specific_indicators",
  "indicator_codes": ["RBC"],
  "category_codes": [],
  "time_scope": "latest",
  "needs_clarification": false
}
```

因此在已经选择患者 `P10086` 后，“红细胞的呢”可以利用 State 中的患者和上一轮查询上下文，
而不需要为“的呢”单独增加生产关键词规则。

## LLM 与 Python 的责任边界

Qwen负责：

- 理解自然语言和多轮省略表达；
- 区分患者查询、能力问题、普通对话和含糊请求；
- 选择单项、类别、全部指标、完整资料和时间范围；
- 普通对话的自然语言回答。

Python负责：

- 从原始用户消息确定性提取患者编号；
- 验证 QueryPlan JSON Schema；
- 验证指标代码和类别白名单；
- 登录身份、角色、医生—患者授权；
- 生成 Tool 参数，禁止模型自由生成 SQL；
- 渲染医疗数据，禁止模型补充诊断和治疗建议；
- Structured Output 解析失败时关闭查询。

## 数据查询能力

本地 SQLite 业务库包含12名合成患者、4次访视和54项指标，主要工具为：

- `query_patient_indicators`：查询指定指标的最新值；
- `query_indicator_trend`：查询指定指标的历史变化；
- `query_patient_full_record`：查询允许展示的基本资料、全部访视和全部指标。

完整资料不会通过普通 `patient:read` 权限返回联系方式和身份证号。

业务数据通过 `MedicalRepository` 接口隔离。当前实现是 SQLite；以后接入 Oracle 时替换
Repository 实现，不需要让模型接触数据库连接或 SQL。

## 认证、角色和管理员

Web 工作台必须登录，浏览器不能提交可信 `actor_id`。服务器使用 HttpOnly Session Cookie、
CSRF Token、scrypt 密码哈希、登录失败限流和审计日志。

本地学习账号：

```text
医生：chen / DemoOnly-2026!
医生：lin  / DemoOnly-2026!
管理员：admin / AdminOnly-2026!
```

这些默认密码只适用于本机合成数据。可以在首次建库前配置：

```text
MEDICAL_AGENT_INITIAL_PASSWORD=医生初始强密码
MEDICAL_AGENT_INITIAL_ADMIN_PASSWORD=不同的管理员强密码
```

管理员登录后进入 `/admin`，可以创建和启停账号、分配角色、分配患者范围以及查看审计。
管理员默认没有患者读取权限。

## 配置与启动

复制 `.env.example` 为 `.env`，至少配置：

```text
DASHSCOPE_API_KEY=你的百炼API密钥
QWEN_MODEL=qwen-plus
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

网页可运行“启动医疗Agent.bat”，或执行：

```powershell
python -m medical_agent.web
```

CLI直接使用Qwen，不再接受 `--mode`：

```powershell
medical-agent "查询患者 P10086 的最新骨密度"
medical-agent --interactive --actor-id doctor-chen
```

## 测试与语义评测

普通测试不会调用外部模型。测试通过 `planner_model` 和 `general_model` 依赖注入假模型，
这属于测试替身，不是可运行的 Demo 模式：

```powershell
pytest
ruff check src tests
```

自然语言泛化不能仅靠单元测试证明。`evals/query_plan_cases.json` 当前包含20类代表性表达，
包括单项、别名、类别、全量、完整资料、趋势、多轮追问、解释、能力问题和否定表达。
配置好Qwen后可运行真实语义评测；该命令会产生实际API调用：

```powershell
python evals/run_query_plan_eval.py
```

评测脚本输出逐项结果和总通过率。以后发现新的语言问题，应先加入对应语义类别和多种改写，
再调整 QueryPlan Prompt 或 Schema，而不是把新短语继续追加到生产 Router。

## 关键文件

```text
src/medical_agent/state.py        QueryPlan 与 LangGraph State
src/medical_agent/planner.py      上下文规划 Prompt、模型调用和白名单解析
src/medical_agent/graph.py        LangGraph 节点与受保护执行路径
src/medical_agent/nodes.py        Authorization、Tool 参数和安全渲染
src/medical_agent/tools.py        医疗数据工具
src/medical_agent/repository.py   存储接口
src/medical_agent/auth.py         登录、角色、会话与审计
evals/                            Qwen真实语义评测集和运行器
```
