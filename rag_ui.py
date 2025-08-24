import streamlit as st
import time
from convai_group_111_rag_vs_ft import RAGConfig,CompleteRAGPipeline,RAGStreamLit
from docx import Document as DocxDocument
from io import BytesIO
from pathlib import Path
import tempfile
import pathlib
def run():
    # Page config
    st.set_page_config(layout="wide")
    st.title("Advanced RAG System with Multi-Stage Retrieval")
    st.caption("This system uses Multi-Stage Retrieval: Stage 1 hybrid search (Dense + BM25), Stage 2 cross-encoder re-ranking. Includes guardrails and performance monitoring.")
    # --- Index Documents ---
    cfg = RAGConfig()
    pipeline = CompleteRAGPipeline(cfg)
    streamLit = RAGStreamLit(pipeline)
    with st.container():
        st.subheader("Index Documents")
        uploaded_files = st.file_uploader("Upload .txt or .docx or .pdf files",accept_multiple_files=True, type=["txt", "docx", "pdf"])
        temp_dir = tempfile.TemporaryDirectory()
        file_paths = []
        if uploaded_files:
            for file in uploaded_files:
                file_name = file.name if file else ''
                st.text(f"{file_name}") 
            
            for uploaded_file in uploaded_files:
                 temp_path = pathlib.Path(temp_dir.name)/ uploaded_file.name
                 with open(temp_path, "wb") as f:
                     f.write(uploaded_file.getbuffer())
                 file_paths.append(temp_path)        
        col1, col2 = st.columns(2)
        with col1:
            chunk_size = st.text_input("Chunk Size (words)", value="300")
        with col2:
            chunk_overlap = st.text_input("Chunk Overlap (words)", value="50")
        guardrails_enabled = st.checkbox("Enable Guardrails", value=True)
        if st.button("Index Documents"):
            with st.spinner("Indexing..."):
                message = streamLit.index_documents(file_paths, chunk_size, chunk_overlap, guardrails_enabled)
                st.success(message)

        with st.expander("Indexing Status", expanded=True):
            st.text("Indexed 1165 files into 792 chunks.")
            st.text("Files Indexed: 1165")
            st.text("Chunks Indexed: 792")

    # --- Ask a Question ---
    with st.container():
        st.subheader("Ask a Question")
        col1, col2 = st.columns([3, 1])

        with col1:
            user_question = st.text_area("Your Question", value="")
        with col2:
            guardrails_enabled_q = st.checkbox("Enable Guardrails", value=True, key="guardrails_q")
            max_docs = st.slider("Max Retrieved Documents", 1, 10, value=5)

        if st.button("Ask"):
            result = streamLit.process_query(user_question,guardrails_enabled_q,max_docs)
            with st.spinner("Retrieving and answering..."):
                time.sleep(3)  # Mocked response time

                st.markdown("#### Answer")
                st.code()

                col3, col4, col5 = st.columns(3)
                col3.metric("Confidence", "1")
                col4.metric("Method Used", "Multi-Stage Retrieval (Hybrid + Cross-Encoder)")
                col5.metric("Response Time", "8.81s")

                with st.expander("Detailed Information"):
                    st.text("Further metadata, logs, or reasoning can be added here.")

