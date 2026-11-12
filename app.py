import streamlit as st
import os
import tempfile
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="我的RAG助手 4.0", page_icon="🚀")
st.title("🚀 我的专属文档问答助手 (全能版)")

# 从后台读取安全 Key
ds_key = st.secrets.get("DEEPSEEK_KEY", "")
sf_key = st.secrets.get("SILICONFLOW_KEY", "")

with st.sidebar:
    st.header("文档上传与管理")
    # 核心升级 1：支持多文件上传
    uploaded_files = st.file_uploader("请上传文档（可多选）", type=["pdf", "txt", "docx"], accept_multiple_files=True)
    api_key = st.text_input("DeepSeek API Key", type="password", value=ds_key)
    
    if st.button("🗑️ 清空当前对话记录"):
        st.session_state.messages = []
        st.rerun()

if uploaded_files and api_key and sf_key:
    # 将多文件转为文本并分块
    @st.cache_resource(show_spinner="正在解析并建立索引...")
    def process_docs(files_content):
        all_splits = []
        for file_name, file_content in files_content:
            ext = file_name.split('.')[-1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp_file:
                tmp_file.write(file_content)
                tmp_path = tmp_file.name
                
            if ext == "pdf":
                loader = PyPDFLoader(tmp_path)
            elif ext == "txt":
                loader = TextLoader(tmp_path, encoding="utf-8")
            elif ext == "docx":
                loader = Docx2txtLoader(tmp_path)
                
            docs = loader.load()
            # 给每个文档打上文件名标签，方便溯源
            for doc in docs:
                doc.metadata["source_file"] = file_name
                
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            splits = text_splitter.split_documents(docs)
            all_splits.extend(splits)

        # 建立向量检索
        my_embedding = OpenAIEmbeddings(
            api_key=sf_key, 
            base_url="https://api.siliconflow.cn/v1", 
            model="BAAI/bge-m3"
        )
        vectorstore = Chroma.from_documents(documents=all_splits, embedding=my_embedding)
        vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # 核心升级 2：建立 BM25 关键词检索
        bm25_retriever = BM25Retriever.from_documents(all_splits)
        bm25_retriever.k = 3

        # 核心升级 3：混合检索（RRF 融合）
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever], weights=[0.5, 0.5]
        )
        return ensemble_retriever

    # 整理文件内容以便传给缓存函数
    files_content = [(f.name, f.getvalue()) for f in uploaded_files]
    ensemble_retriever = process_docs(tuple(files_content))

    st.success(f"成功加载 {len(uploaded_files)} 个文件！你可以开始提问了。")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("针对这些文档，你想问什么？"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            llm = ChatOpenAI(model="deepseek-chat", temperature=0, api_key=api_key, base_url="https://api.deepseek.com/v1")
            
            # 查询重写（3.0 核心）
            if len(st.session_state.messages) > 1:
                with st.spinner("🤔 正在结合上下文分析问题..."):
                    rewrite_prompt = f"""根据以下对话历史，将用户的最新输入重写为一个独立的、完整的、包含所有上下文信息的问题。只输出重写后的问题。
对话历史：{st.session_state.messages[-3:-1]}
最新输入：{prompt}
重写后的问题："""
                    rewritten_response = llm.invoke(rewrite_prompt)
                    search_query = rewritten_response.content.strip()
                    st.caption(f"🔄 系统结合上下文重写的问题：*{search_query}*")
            else:
                search_query = prompt

            # 使用混合检索
            retrieved_docs = ensemble_retriever.invoke(search_query)
            context = "\n".join([d.page_content for d in retrieved_docs])

            system_prompt = f"""你是一个专业的文档分析助手。请严格根据以下提供的上下文内容回答用户的问题。
如果你在上下文中找不到答案，请直接说“根据提供的文档，我无法回答这个问题”，绝对不要自己编造。
上下文内容：
{context}
"""
            messages = [{"role": "system", "content": system_prompt}]
            for msg in st.session_state.messages:
                messages.append({"role": msg["role"], "content": msg["content"]})

            response = llm.stream(messages)
            response_text = st.write_stream(response)
            
            # 引用溯源（现在带文件名了）
            with st.expander("🔍 查看引用的原文档片段"):
                for i, doc in enumerate(retrieved_docs):
                    source = doc.metadata.get('source_file', '未知文件')
                    st.info(f"来自文件：{source} \n\n{doc.page_content}")
            
        st.session_state.messages.append({"role": "assistant", "content": response_text})
else:
    st.info("👈 请在左侧上传文档。确保后台已配置好 API Key。")
