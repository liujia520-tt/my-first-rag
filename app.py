import streamlit as st
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import tempfile

st.set_page_config(page_title="我的RAG助手", page_icon="🤖")
st.title("🤖 我的专属文档问答助手")

# 🌟 核心修改：从 Streamlit Cloud 的安全后台读取真正的 Key
ds_key = st.secrets.get("DEEPSEEK_KEY", "")
sf_key = st.secrets.get("SILICONFLOW_KEY", "")

with st.sidebar:
    st.header("文档上传")
    uploaded_file = st.file_uploader("请上传文档", type=["pdf", "txt", "docx"])
    # 侧边栏依然保留输入框，默认填入从后台读取的 Key
    api_key = st.text_input("DeepSeek API Key", type="password", value=ds_key)

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
        
        # 🌟 这里直接使用从后台读取的硅基流动 Key
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

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("针对这份文档，你想问什么？"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            retriever = init_rag(tmp_path, file_extension)
            retrieved_docs = retriever.invoke(prompt)
            context = "\n".join([d.page_content for d in retrieved_docs])

            llm = ChatOpenAI(model="deepseek-chat", temperature=0, api_key=api_key, base_url="https://api.deepseek.com/v1")
            prompt_template = f"根据以下内容回答问题：\n内容：{context}\n问题：{prompt}"
            
            response = llm.stream(prompt_template)
            response_text = st.write_stream(response)
            
        st.session_state.messages.append({"role": "assistant", "content": response_text})
else:
    st.info("👈 请在左侧上传文档。确保后台已配置好 API Key。")
