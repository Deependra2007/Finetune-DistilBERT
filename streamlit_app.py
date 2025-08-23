import streamlit as st
import rag_ui
import Finetune

st.title("Choose Model for Execution")
# Initialize session state
if "model" not in st.session_state:
    st.session_state.model = None
option = st.selectbox(
    "Select an Option",
    ["Select...", "RAG", "Fine Tune"]
)

if option is not None:
    st.session_state.model = option
if st.button("Execute") or st.session_state.file_uploaded is not None:
    if st.session_state.model == "RAG":
        rag_ui.run()
    elif st.session_state.model == "Fine Tune":
        Finetune.run()
    else:
        st.warning("Please select a valid option before clicking Run.")
