import streamlit as st
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, AIMessage
import tempfile

st.set_page_config(page_title="我的RAG助手 2.0", page_icon="🚀")
st.title("🚀 我的专属文档问答助手 (进阶版)")

# 从后台读取安全 Key
ds_key = st.secrets.get("DEEPSEEK_KEY", "")
sf_key = st.secrets.get("SILICONFLOW_KEY", "")

# 侧边栏设置
with st.sidebar:
    st.header("文档上传与管理")
    uploaded_file = st.file_uploader("请上传文档", type=["pdf", "txt", "docx"])
    api_key = st.text_input("DeepSeek API Key", type="password", value=ds_key)
    
    # 添加一个清空对话的按钮
    if st.button("🗑️ 清空当前对话记录"):
        st.session_state.messages = []
        st.rerun()

if uploaded_file is not None and api_key and sf_key:
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    # 处理文档并建立向量库
    @st.cache_resource
    def init_rag(file_path, ext):
        if ext == "pdf":
            loader = PyPDFLoader(file_path)
        elif ext == "txt":
            loader = TextLoader(file_path, encoding="utf-8")
        elif ext == "docx":
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError("不支持的文件格式")
            
        docs = loader.load()
        # chunk_size设为500，overlap设为50保持语义连贯
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = text_splitter.split_documents(docs)
        
        my_embedding = OpenAIEmbeddings(
            api_key=sf_key, 
            base_url="https://api.siliconflow.cn/v1", 
            model="BAAI/bge-m3"
        )
        vectorstore = Chroma.from_documents(documents=splits, embedding=my_embedding)
        return vectorstore.as_retriever()

    st.success("文档加载成功！你可以开始提问了。")

    # 初始化对话历史（多轮记忆）
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 展示历史对话
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 接收用户输入
    if prompt := st.chat_input("针对这份文档，你想问什么？"):
        # 1. 把用户的问题加入历史
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            retriever = init_rag(tmp_path, file_extension)
            
            # 2. 检索相关文档片段
            retrieved_docs = retriever.invoke(prompt)
            context = "\n".join([d.page_content for d in retrieved_docs])

            # 3. 构建带历史记忆的消息列表发给大模型
            llm = ChatOpenAI(model="deepseek-chat", temperature=0, api_key=api_key, base_url="https://api.deepseek.com/v1")
            
            # 构造带上下文的系统提示词
            system_prompt = f"""你是一个专业的文档分析助手。请严格根据以下提供的上下文内容回答用户的问题。
如果你在上下文中找不到答案，请直接说“根据提供的文档，我无法回答这个问题”，绝对不要自己编造。
            
上下文内容：
{context}
"""
            # 将历史记录转换为 LangChain 的消息格式
            messages = [{"role": "system", "content": system_prompt}]
            for msg in st.session_state.messages:
                messages.append({"role": msg["role"], "content": msg["content"]})

            # 4. 流式输出回答
            response = llm.stream(messages)
            response_text = st.write_stream(response)
            
            # 5. 实现引用溯源（把检索到的原文档片段折叠显示）
            with st.expander("🔍 查看引用的原文档片段"):
                for i, doc in enumerate(retrieved_docs):
                    st.info(f"片段 {i+1}：\n{doc.page_content}")
            
        # 6. 把 AI 的回答也加入历史记录
        st.session_state.messages.append({"role": "assistant", "content": response_text})
else:
    st.info("👈 请在左侧上传文档。确保后台已配置好 API Key。")
