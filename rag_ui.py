import streamlit as st
import time
from convai_group_111_rag_vs_ft import RAGConfig, CompleteRAGPipeline
from pathlib import Path
import tempfile

# Use Streamlit's caching to load the pipeline only once
@st.cache_resource
def load_pipeline():
    """Loads and initializes the RAG pipeline."""
    # The RAGConfig now comes from your backend file
    cfg = RAGConfig()
    return CompleteRAGPipeline(cfg)

def run():
    # --- Page Configuration ---
    st.set_page_config(layout="wide")
    st.title("Advanced RAG System with Multi-Stage Retrieval")
    st.caption("This system uses Multi-Stage Retrieval: Stage 1 hybrid search (Dense + BM25), Stage 2 cross-encoder re-ranking.")

    # --- Load Pipeline & Initialize Session State ---
    pipeline = load_pipeline()
    if 'indexing_info' not in st.session_state:
        st.session_state.indexing_info = None

    # --- 1. Index Documents ---
    with st.container():
        st.subheader("Index Documents")
        uploaded_files = st.file_uploader(
            "Upload .txt or .docx or .pdf files  or Zip file with csv",
            accept_multiple_files=True,
            type=["txt", "docx", "pdf","zip"]
        )
        
        col1, col2 = st.columns(2)
        with col1:
            chunk_size = st.text_input("Chunk Size (words)", value="300")
        with col2:
            chunk_overlap = st.text_input("Chunk Overlap (words)", value="50")
        
        # This checkbox is for display; the actual logic is in your backend
        guardrails_enabled = st.checkbox("Enable Guardrails", value=True)

        if st.button("Index Documents"):
            if uploaded_files:
                with st.spinner("Processing and indexing documents... This may take a moment."):
                    # Use a temporary directory that is automatically cleaned up
                    with tempfile.TemporaryDirectory() as temp_dir:
                        file_paths = []
                        for uploaded_file in uploaded_files:
                            temp_path = Path(temp_dir) / uploaded_file.name
                            with open(temp_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            file_paths.append(temp_path)
                        
                        try:
                            # Call the new run_indexing method directly
                            indexing_result = pipeline.run_indexing(
                                file_paths, int(chunk_size), int(chunk_overlap)
                            )
                            # Store the results in the session state
                            st.session_state.indexing_info = indexing_result
                            st.success(st.session_state.indexing_info['message'])
                        except Exception as e:
                            st.error(f"An error occurred during indexing: {e}")
            else:
                st.warning("Please upload files to index.")

        # --- DYNAMIC Indexing Status Expander ---
        if st.session_state.indexing_info:
            with st.expander("Indexing Status", expanded=True):
                info = st.session_state.indexing_info
                col1, col2 = st.columns(2)
                col1.metric("Files Indexed", info.get('files_indexed', 0))
                col2.metric("Chunks Created", info.get('chunks_created', 0))

    # --- 2. Ask a Question ---
    # This section now only appears AFTER documents have been indexed successfully
    if st.session_state.indexing_info:
        with st.container():
            st.subheader("Ask a Question")
            q_col1, q_col2 = st.columns([3, 1])

            with q_col1:
                user_question = st.text_area("Your Question", placeholder="What were Microsoft's diluted earnings per share in 2023?")
            with q_col2:
                guardrails_enabled_q = st.checkbox("Enable Guardrails", value=True, key="guardrails_q")
                max_docs = st.slider("Max Retrieved Documents", 1, 10, value=5)

            if st.button("Ask"):
                if user_question:
                    with st.spinner("Retrieving and answering..."):
                        # Call the pipeline's query method to get a real answer
                        response = pipeline.query(user_question)

                        if response.get('success'):
                            st.markdown("#### Answer")
                            st.info(response.get('answer', 'No answer could be generated.'))

                            res_col1, res_col2, res_col3 = st.columns(3)
                            res_col1.metric("Confidence", f"{response.get('confidence', 0.0):.2f}")
                            res_col2.metric("Method Used", response.get('method', 'RAG'))
                            res_col3.metric("Response Time", f"{response.get('response_time', 0.0):.2f}s")

                            with st.expander("Detailed Information & Retrieved Documents"):
                                st.json(response.get('retrieval_details', {}))
                        else:
                            st.error(f"Failed to get an answer: {response.get('error', 'Unknown error')}")
                else:
                    st.warning("Please enter a question.")
