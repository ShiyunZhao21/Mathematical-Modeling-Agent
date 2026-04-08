<h1 align="center">🤖 MathModelAgent 📐</h1>
<p align="center">
    <img src="./docs/icon.png" height="250px">
</p>
<h4 align="center">
    专为数学建模设计的 Agent<br>
    自动完成数学建模，生成一份完整的可以直接提交的论文。
</h4>

<h5 align="center">简体中文 | <a href="README_EN.md">English</a></h5>

## 🌟 愿景：

3 天的比赛时间变为 1 小时 <br> 
自动完整一份可以获奖级别的建模论文

<p align="center">
    <img src="./docs/index.png">
    <img src="./docs/chat.png">
    <img src="./docs/coder.png">
    <img src="./docs/writer.png">
</p>

## ✨ 功能特性

- 🔍 自动分析问题，数学建模，编写代码，纠正错误，撰写论文
- 💻 Code Interpreter
    - local Interpreter: 基于 jupyter , 代码保存为 notebook 方便再编辑
    - 云端 code interpreter: [E2B](https://e2b.dev/) 和 [daytona](https://app.daytona.io/)
- 📝 生成一份编排好格式的论文
- 🤝 multi-agents: 建模手，代码手，论文手等
- 🔄 multi-llms: 每个 agent 设置不同的、合适的模型
- 🤖 支持所有模型: [litellm](https://docs.litellm.ai/docs/providers)
- 💰 成本低：workflow agentless，不依赖 agent 框架
- 🧩 自定义模板：prompt inject 为每个 subtask 单独设置需求

## 🚀 后期计划

- [x] 添加并完成 webui、cli
- [ ] 完善的教程、文档
- [ ] 提供 web 服务
- [ ] 英文支持（美赛）
- [ ] 集成 latex 模板
- [ ] 接入视觉模型
- [x] 添加正确文献引用
- [x] 更多测试案例
- [x] docker 部署
- [ ] human in loop ( HIL ): 引入用户的交互（选择模型，@agent重写，等等）
- [ ] feedback: evaluate the result and modify
- [x] codeinterpreter 接入云端 如 e2b 等供应商..
- [ ] 多语言: R 语言, matlab
- [ ] 绘图 napki,draw.io,plantuml,svg, mermaid.js
- [ ] 添加 benchmark
- [ ] web search tool
- [ ] RAG 知识库
- [ ] A2A hand off: 代码手多次反思错误，hand off 更聪明模型 agent
- [ ] chat / agent mode

## 视频demo

<video src="https://github.com/user-attachments/assets/954cb607-8e7e-45c6-8b15-f85e204a0c5d"></video>

> [!CAUTION]
> 项目处于实验探索迭代demo阶段，有许多需要改进优化改进地方，我(项目作者)很忙，有时间会优化更新
> 欢迎贡献


## 📖 使用教程

当前仓库已经收缩为**纯后端版本**，保留了多 Agent 建模流水线，不再包含前端界面、WebSocket 和旧实时路由。

### 🐳 方案一：Docker 部署

```bash
git clone https://github.com/jihe520/MathModelAgent.git
cd MathModelAgent
docker-compose up --build
```

启动后服务监听：

- 后端 API: `http://localhost:8000`
- 健康检查: `http://localhost:8000/health`

### 💻 方案二：本地部署

> 只需要 Python 环境，不再依赖 Node.js、Redis 或前端工程

```bash
cd backend
pip install uv
uv sync
```

启动后端：

```bash
cd backend
source .venv/bin/activate # MacOS / Linux
venv\Scripts\activate.bat # Windows
ENV=DEV uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 提交任务

使用 `multipart/form-data` 调用：

- `POST /tasks/run`
- `ques_all`: 完整赛题文本
- `comp_template`: 默认 `CHINA`
- `format_output`: 当前仅支持 `Markdown`
- `files`: 可选数据文件

可继续查询：

- `GET /tasks/{task_id}`：任务状态
- `GET /tasks/{task_id}/messages`：过程消息
- `GET /tasks/{task_id}/artifacts`：输出文件

### 输出目录

运行结果保存在 `backend/project/work_dir/<task_id>/` 目录下，典型产物包括：

- `paper.md`
- `paper.docx`
- `paper.pdf`
- `solution.ipynb`
- `solution.py`
- 过程图片和中间文件

需要自定义提示词模板：
Prompt Inject : [prompt](./backend/app/config/md_template.toml)

网络状况太差难以配置 Docker 等设置？
网络不畅时的配置过程示例：[网络环境极差时的MathModelAgent配置过程](docs/md/网络环境极差时的MathModelAgent配置过程.md)


## 🤝 贡献和开发

[DeepWiki](https://deepwiki.com/jihe520/MathModelAgent) | [Zread](https://zread.ai/jihe520/MathModelAgent)


> [!TIP]
> 如果你有跑出来好的案例可以提交 PR 在该仓库下:
> [MathModelAgent-Example](https://github.com/jihe520/MathModelAgent-Example)

- 项目处于**开发实验阶段**（我有时间就会更新），变更较多，还存在许多 Bug，我正着手修复。
- 希望大家一起参与，让这个项目变得更好
- 非常欢迎使用和提交  **PRs** 和 issues 
- 需求参考 后期计划

clone 项目后，下载 **Todo Tree** 插件，可以查看代码中所有具体位置的 todo

`.cursor/*` 有项目整体架构、rules、mcp 可以方便开发使用

## 📄 版权License

个人免费使用，请勿商业用途，商业用途联系我（作者）

[License](./docs/md/License.md)

## 🙏 Reference

Thanks to the following projects:
- [OpenCodeInterpreter](https://github.com/OpenCodeInterpreter/OpenCodeInterpreter/tree/main)
- [TaskWeaver](https://github.com/microsoft/TaskWeaver)
- [Code-Interpreter](https://github.com/MrGreyfun/Local-Code-Interpreter/tree/main)
- [Latex](https://github.com/Veni222987/MathModelingLatexTemplate/tree/main)
- [Agent Laboratory](https://github.com/SamuelSchmidgall/AgentLaboratory)
- [ai-manus](https://github.com/Simpleyyt/ai-manus)

## 其他

### 💖 Sponsor

[☕️ 给作者买一杯咖啡](./docs/md/sponser.md)

感谢赞助

#### 企业

[![Powered by DartNode](https://dartnode.com/branding/DN-Open-Source-sm.png)](https://dartnode.com "Powered by DartNode - Free VPS for Open Source")

<div align="center">
    <a href="https://share.302.ai/UoTruU" target="_blank">
    <img src="./docs/302ai.jpg">
    </a>
</div>

[302.AI](https://share.302.ai/UoTruU) 是一个按用量付费的企业级AI资源平台，提供市场上最新、最全面的AI模型和API，以及多种开箱即用的在线AI应用

#### 用户

[danmo-tyc](https://github.com/danmo-tyc)

### 👥 GROUP

有问题可以进群问

点击链接加入腾讯频道【MathModelAgent】：https://pd.qq.com/s/7rfbai3au

点击链接加入群聊 779159301【MathModelAgent】：https://qm.qq.com/q/Fw2cCJPoki

[Discord](https://discord.gg/3Jmpqg5J)

> [!CAUTION]
> 免责声明: 注意，AI 生成仅供参考，目前水平直接参加国赛获奖是不可能的，但我相信 AI 和 该项目未来的成长。
