import streamlit as st
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import tempfile

st.set_page_config(page_title="我的RAG助手 3.0", page_icon="🚀")
st.title("🚀 我的专属文档问答助手 (终极进化版)")

# 从后台读取安全 Key
ds_key = st.secrets.get("DEEPSEEK_KEY", "")
sf_key = st.secrets.get("SILICONFLOW_KEY", "")

with st.sidebar:
    st.header("文档上传与管理")
    uploaded_file = st.file_uploader("请上传文档", type=["pdf", "txt", "docx"])
    api_key = st.text_input("DeepSeek API Key", type="password", value=ds_key)
    
    if st.button("🗑️ 清空当前对话记录"):
        st.session_state.messages = []
        st.rerun()

if uploaded_file is not None and api_key and sf_key:
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

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

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 展示历史对话
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("针对这份文档，你想问什么？"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # 初始化 LLM
            llm = ChatOpenAI(model="deepseek-chat", temperature=0, api_key=api_key, base_url="https://api.deepseek.com/v1")
            
            # ================= 核心 3.0 逻辑：查询重写 =================
            # 如果历史记录超过1条（说明不是第一句话），则进行重写
            if len(st.session_state.messages) > 1:
                with st.spinner("🤔 正在结合上下文分析问题..."):
                    rewrite_prompt = f"""根据以下对话历史，将用户的最新输入重写为一个独立的、完整的、包含所有上下文信息的问题。
如果最新输入已经很完整，则保持原样。只输出重写后的问题，不要输出任何解释。
                    
对话历史：
{st.session_state.messages[-3:-1]}
                    
最新输入：{prompt}
重写后的问题："""
                    rewritten_response = llm.invoke(rewrite_prompt)
                    search_query = rewritten_response.content.strip()
                    st.caption(f"🔄 系统结合上下文重写的问题：*{search_query}*")
            else:
                search_query = prompt
            # ==========================================================

            # 使用重写后的完整问题去检索
            retriever = init_rag(tmp_path, file_extension)
            retrieved_docs = retriever.invoke(search_query)
            context = "\n".join([d.page_content for d in retrieved_docs])

            system_prompt = f"""你是一个专业的文档分析助手。请严格根据以下提供的上下文内容回答用户的问题。
如果你在上下文中找不到答案，请直接说“根据提供的文档，我无法回答这个问题”，绝对不要自己编造。
            
上下文内容：
{context}
"""
            messages = [{"role": "system", "content": system_prompt}]
            for msg in st.session_state.messages:
                messages.append({"role": msg["role"], "content": msg["content"]})

            # 流式输出
            response = llm.stream(messages)
            response_text = st.write_stream(response)
            
            # 引用溯源
            with st.expander("🔍 查看引用的原文档片段"):
                for i, doc in enumerate(retrieved_docs):
                    st.info(f"片段 {i+1}：\n{doc.page_content}")
            
        st.session_state.messages.append({"role": "assistant", "content": response_text})
else:
    st.info("👈 请在左侧上传文档。确保后台已配置好 API Key。")
