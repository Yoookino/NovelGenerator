# 小说创作与续写助手

一个基于大模型的命令行小说创作工具。它把长篇创作拆成了一条可重复执行的流水线：

- 生成小说蓝图与基础设定
- 为单章生成写作计划
- 按分段写出章节正文
- 自动做章节质检与必要修订
- 归档摘要、人物状态、设定和伏笔记忆

当前项目的核心实现位于 [novel_writer.py](./novel_writer.py)。

## 功能概览

- `小说蓝图生成`：输入一句创意，自动生成书名、题材、卖点、世界规则、角色卡和卷规划
- `章节计划生成`：先规划单章目标、冲突、反转、钩子和分段安排，再开始写正文
- `分段写作`：按章节计划逐段生成，减少长文本一次性失控的问题
- `自动质检与修订`：对章节做结构、节奏、人设、设定一致性检查，并按需要修订
- `结构化记忆`：把剧情、伏笔、人物变更、设定更新写入 JSON 和记忆向量库
- `多模型切换`：支持在 `Grok` 和 `DeepSeek` 之间切换
- `章节润色`：对已存在章节进行文风优化

## 项目结构

```text
Novel/
├─ novel_writer.py          # 主程序，CLI 入口
├─ README.md                # 项目说明
├─ .env                     # 本地环境变量（不要提交真实密钥）
├─ data/                    # 运行数据目录
│  ├─ chapters/             # 章节正文与润色稿
│  ├─ summaries/            # 章节结构化摘要
│  ├─ outlines/             # 小说总纲、卷纲、蓝图
│  ├─ prompts/              # 风格配置
│  ├─ settings/             # 角色设定与人物状态
│  ├─ plans/                # 单章写作计划
│  ├─ qa_reports/           # 章节质检报告
│  ├─ memory/               # 结构化剧情/设定/伏笔记忆
│  └─ vector_store.npy      # 向量记忆库
└─ eronovel/                # 本地虚拟环境目录
```

## 环境要求

- Python 3.10+
- 可访问对应模型 API
- 建议准备代理环境，尤其是访问海外模型接口时

依赖以当前代码为准，核心使用到：

- `requests`
- `numpy`
- `python-dotenv`

如果你还没有安装依赖，可以先执行：

```bash
pip install requests numpy python-dotenv
```

## 配置说明

程序通过 `.env` 读取配置。

示例：

```env
HTTP_PROXY=http://127.0.0.1:7890
HTTPS_PROXY=http://127.0.0.1:7890

SILICONFLOW_API_KEY=your_siliconflow_key
DEEPSEEK_API_KEY=your_deepseek_key
GROK_API_KEY=your_grok_key

EMBED_MODEL=BAAI/bge-m3
DEEPSEEK_MODEL=deepseek-chat
GROK_MODEL=grok-4-1-fast-reasoning
LLM_PROVIDER=grok
```

说明：

- `SILICONFLOW_API_KEY` 用于向量化检索
- `DEEPSEEK_API_KEY` 用于 DeepSeek 对话模型
- `GROK_API_KEY` 用于 Grok 对话模型
- `LLM_PROVIDER` 默认模型提供方，可设为 `grok` 或 `deepseek`

## 如何启动

```bash
python novel_writer.py
```

启动后可在命令行中输入指令。

## CLI 指令

### 框架搭建

- `/outline <idea>`：根据一句话创意生成小说蓝图、大纲、角色和文风配置
- `/blueprint`：查看当前小说蓝图
- `/state`：查看当前人物状态卡
- `/style`：查看当前文风配置

### 章节创作

- `/plan <n>`：只生成第 `n` 章的写作计划
- `/new <n>`：生成第 `n` 章
- `/next`：自动续写下一章
- `/auto <n>`：连续自动生成 `n` 章
- `/polish <n>`：润色第 `n` 章

### 内容查看

- `/show <n>`：查看第 `n` 章正文
- `/showplan <n>`：查看第 `n` 章计划
- `/showqa <n>`：查看第 `n` 章质检报告
- `/stat`：查看已完成章节数、累计字数和本次运行 Token 统计

### 系统命令

- `/help`：显示帮助
- `/switch`：切换当前模型提供方
- `/exit`：退出程序

## 典型使用流程

### 1. 从创意开始搭建小说

```text
/outline 现代都市背景下，一名失忆调查员追查连环梦境杀人案
```

这一步会生成并写入：

- `data/outlines/novel_blueprint.json`
- `data/outlines/novel_outline.txt`
- `data/outlines/arc_outline.txt`
- `data/settings/characters.txt`
- `data/prompts/style_prompt.txt`
- `data/settings/character_state.json`

### 2. 先做单章计划

```text
/plan 1
```

生成结果会写入：

- `data/plans/chapter_001_plan.json`
- `data/plans/chapter_001_plan.txt`

### 3. 生成正文并自动归档

```text
/new 1
```

这一条命令会依次执行：

1. 生成章节计划
2. 分段写作
3. 自动质检
4. 按需修订
5. 写入摘要与结构化记忆

### 4. 继续续写

```text
/next
/auto 3
```

### 5. 润色已有章节

```text
/polish 1
```

润色结果会输出到：

- `data/chapters/chapter_001_polished.txt`

## 数据文件说明

### 蓝图与大纲

- `data/outlines/novel_blueprint.json`：结构化小说蓝图
- `data/outlines/novel_outline.txt`：总纲文本
- `data/outlines/arc_outline.txt`：卷纲/阶段规划

### 风格与角色

- `data/prompts/style_profile.json`：结构化文风配置
- `data/prompts/style_prompt.txt`：可读文本风格说明
- `data/settings/characters.txt`：角色卡文本版
- `data/settings/character_state.json`：动态人物状态

### 创作过程产物

- `data/plans/chapter_xxx_plan.*`：章节计划
- `data/chapters/chapter_xxx.txt`：章节正文
- `data/qa_reports/chapter_xxx_qa.*`：章节质检
- `data/summaries/chapter_xxx_summary.txt`：高密度摘要

### 长期记忆

- `data/memory/plot_memory.json`：剧情记忆
- `data/memory/setting_memory.json`：设定记忆
- `data/memory/foreshadow_memory.json`：伏笔记忆
- `data/vector_store.npy`：向量检索库

## 当前实现特点

- 当前是 `单文件核心实现`，便于快速迭代，但后续适合拆模块
- 使用 `requests + SSE 流式输出` 直接调用对话模型
- 向量检索使用 `numpy` 存储本地数组，适合轻量项目，不适合大规模生产
- 所有数据直接落地到 `data/`，便于手工查看和调试

## 已知问题与建议

- 旧 README 曾出现编码乱码，现已重写
- `.env` 中应只保留本地测试密钥，避免提交真实凭据
- `eronovel/` 是本地虚拟环境，通常建议加入忽略列表而不是提交到仓库
- 当前没有独立测试文件，后续建议补充最少量的单元测试和集成测试
- `novel_writer.py` 体积已经较大，建议后续拆分为 `llm`、`memory`、`planner`、`writer`、`cli` 等模块

## 安全提醒

- 本项目会调用第三方模型 API，注意额度和费用
- 生成内容请自行审阅，尤其是公开发布前
- 不要在仓库中提交真实 API Key、代理账号或其他敏感配置


## 许可证

如果你准备公开发布，建议补充明确的许可证文件。
