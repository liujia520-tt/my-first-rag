# 🚀 我的专属 RAG 文档问答助手

基于 LangChain 与大语言模型（LLM）构建的企业级文档问答系统。支持上传本地文档，通过检索增强生成（RAG）技术，实现精准、带溯源的智能问答。

🔗 **在线体验 (Live Demo)**: [点击这里访问我的助手](https://my-first-rag-cqwxhdmwpsf3bmso3wqnvh.streamlit.app)

## ✨ 核心功能
- **多格式文档解析**：支持 PDF、TXT、Word (Docx) 等多种格式的文档读取与加载。
- **BGE-M3 向量化**：接入 SiliconFlow 平台的 `BAAI/bge-m3` 模型，将文本块转换为高维向量。
- **ChromaDB 向量存储**：使用本地 Chroma 向量数据库进行文档片段的持久化存储与高效相似度检索。
- **多轮对话记忆**：基于 Streamlit Session State 结合 LangChain，实现带历史上下文的连续对话。
- **引用溯源机制**：回答下方附带“查看引用的原文档片段”折叠框，杜绝大模型幻觉，让回答有据可查。
- **流式输出**：结合 DeepSeek API，实现打字机效果的实时流式回答。

## 🛠️ 技术栈
- **前端 UI**: Streamlit
- **核心框架**: LangChain
- **大语言模型**: DeepSeek (deepseek-chat)
- **Embedding 模型**: SiliconFlow (BAAI/bge-m3)
- **向量数据库**: ChromaDB

## 🚀 本地运行指南
1. 克隆项目：
   ```bash
   git clone https://github.com/liujia520-tt/my-first-rag.git
