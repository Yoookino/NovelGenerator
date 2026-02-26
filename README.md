# 小说创作与润色助手

一个基于AI的小说创作工具，支持章节生成、润色、摘要提取和自动续写功能。

## 功能特性

- **章节生成**：根据大纲自动撰写小说章节
- **智能润色**：优化文笔，增强氛围感
- **自动续写**：智能识别进度，连续生成后续章节
- **记忆系统**：向量检索过往情节，保持连贯性
- **双模型支持**：可在Grok和DeepSeek之间切换
- **摘要提取**：自动生成高密度剧情摘要

## 项目结构

```
your-project/
├── main.py              # 主程序文件
├── requirements.txt     # 依赖包列表
├── README.md           # 说明文档
└── data/               # 数据目录（运行后自动创建）
    ├── chapters/       # 章节文件
    ├── summaries/      # 摘要文件
    ├── prompts/        # 提示词模板
    ├── outlines/       # 大纲文件
    ├── settings/       # 设定文件
    └── vector_store.npy # 向量记忆库
```

## 安装与配置

### 1. 环境要求
- Python 3.8+
- 网络连接（用于API调用）

### 2. 安装依赖
```bash
# 克隆项目（如果使用Git）
git clone <你的仓库地址>
cd <项目目录>

# 安装依赖包
pip install -r requirements.txt
```

如果没有 `requirements.txt`，可以手动安装：
```bash
pip install requests numpy
```

### 3. API配置
1. 获取API密钥：
   - **SiliconFlow API**：用于文本向量化，[注册获取](https://siliconflow.cn/)
   - **DeepSeek API**：备用LLM模型，[注册获取](https://platform.deepseek.com/)
   - **Grok API**：主LLM模型（需要特殊权限）

2. 修改配置文件：
   在 `main.py` 第16-18行，替换为你的API密钥：
   ```python
   SILICONFLOW_API_KEY = "你的_siliconflow_api_key"
   DEEPSEEK_API_KEY = "你的_deepseek_api_key"
   GROK_API_KEY = "你的_grok_api_key"
   ```

### 4. 代理设置（可选）
如果需要代理，请修改第10-11行：
```python
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'  # 你的代理地址
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
```

## 准备工作

### 1. 创建必要文件
在运行前，建议先在 `data/` 目录下创建以下文件：

**data/outlines/novel_outline.txt** - 小说总纲
```
这是一个关于星际探险的故事。主角林风意外获得神秘晶体，开启穿梭不同星球的能力...
```

**data/outlines/arc_outline.txt** - 当前卷大纲
```
第一卷：觉醒之路
第1-5章：发现神秘晶体，初步掌握能力
第6-10章：第一次星际穿越，遭遇外星文明...
```

**data/settings/characters.txt** - 角色设定
```
林风：23岁，考古学研究生，冷静谨慎，有冒险精神...
苏雅：25岁，外星生物学家，热情开朗，知识渊博...
```

**data/prompts/style_prompt.txt** - 风格提示
```
写作风格：中等节奏，注重环境描写和心理活动，对话自然
网文要素：适当加入悬念和伏笔，每章结尾留钩子
```

### 2. 初始化第一章（可选）
如果从第一章开始，可以手动创建：
```bash
echo "这是第一章的内容..." > data/chapters/chapter_001.txt
```

## 使用指南

### 启动程序
```bash
python main.py
```

### 可用命令

| 命令 | 功能 | 示例 |
|------|------|------|
| `/help` | 显示帮助信息 | `/help` |
| `/new <n>` | 生成第n章 | `/new 5` |
| `/next` | 自动续写下一章 | `/next` |
| `/auto <n>` | 连续生成n章 | `/auto 3` |
| `/polish <n>` | 润色第n章 | `/polish 2` |
| `/show <n>` | 显示第n章内容 | `/show 1` |
| `/stat` | 查看统计信息 | `/stat` |
| `/switch` | 切换AI模型 | `/switch` |
| `/exit` | 退出程序 | `/exit` |

### 工作流程示例

#### 1. 开始创作新小说
```bash
# 启动程序
python main.py

# 生成第一章
/new 1

# 查看生成的内容
/show 1

# 继续生成后续章节
/next
/next

# 或者批量生成
/auto 5
```

#### 2. 润色已写章节
```bash
# 润色第3章
/polish 3

# 润色后的文件会保存为 chapter_003_polished.txt
```

#### 3. 查看统计数据
```bash
# 查看创作进度
/stat

输出示例：
[统计情报]
- 已完成章节: 8 章
- 累计总字数: 24567 字
- 本次运行 Token 消耗: 12450
```

## 🔄 模型切换

程序默认使用Grok模型，如需切换到DeepSeek：
```bash
# 切换模型
/switch
# 系统提示：--- 模型已切换为: deepseek ---
```

## 文件说明

### 生成的文件
- **章节文件**：`chapter_001.txt`, `chapter_002.txt`, ...
- **润色文件**：`chapter_001_polished.txt`, ...
- **摘要文件**：`chapter_001_summary.txt`, ...
- **记忆库**：`vector_store.npy`（自动生成和维护）

### 摘要文件格式
每个章节的摘要包含：
```
【核心剧情】：...
【关键冲突】：...
【伏笔/信息】：...
【状态变更】：...
```

## 注意事项

1. **API费用**：使用AI服务会产生费用，请关注API使用量
2. **网络稳定性**：程序有重试机制，但网络不佳时可能失败
3. **内容审核**：生成的内容需符合平台规范
4. **备份重要数据**：定期备份 `data/` 目录下的文件

## 🔍 故障排除

### 常见问题

1. **API调用失败**
   - 检查API密钥是否正确
   - 确认网络连接正常
   - 查看代理设置（如有需要）

2. **找不到文件**
   - 确认文件路径和名称正确
   - 检查文件编码（支持UTF-8和GBK）

3. **内存错误**
   - 确保有足够的磁盘空间
   - 如果 `vector_store.npy` 损坏，可以删除后重启程序

4. **章节编号混乱**
   - 程序会自动检测最高编号的章节
   - 可以手动整理 `chapters/` 目录下的文件

### 日志解读
程序运行时显示的信息：
- `[Tokens] 输入: X | 输出: Y | 消耗: Z` - API使用统计
- `[文件写入]` - 文件保存成功
- `[记忆]` - 摘要已添加到记忆库
- `[网络抖动]` - 网络不稳定，正在重试

## 性能优化建议

1. **批量操作**：使用 `/auto` 命令批量生成，比单章生成效率高
2. **合理使用润色**：不需要每章都润色，关键章节润色即可
3. **定期清理**：如果 `vector_store.npy` 过大，可以备份后删除重建
4. **模型选择**：根据需求选择合适的模型（Grok适合创意，DeepSeek成本较低）

## 贡献与反馈

欢迎提交Issue和Pull Request来改进这个项目！

## 许可证

本项目仅供学习和研究使用，请遵守相关AI服务的使用条款。