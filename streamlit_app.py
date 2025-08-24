import streamlit as st
import rag_ui
import Finetune

st.title("Choose Model for Execution")
# Initialize session state
#if "model" not in st.session_state:
   # st.session_state.model = None
option = st.selectbox(
    "Select an Option",
    ["Select...", "RAG", "Fine Tune"]
)

#if option is not None:
  #  st.session_state.model = option
if st.button("Execute"):
    if option == "RAG":
        rag_ui.run()
    elif option == "Fine Tune":
        Finetune.run()
    else:
        st.warning("Please select a valid option before clicking Run.")
