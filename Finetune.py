import streamlit as st
from transformers import pipeline

# Initialize QA pipeline
@st.cache_resource
def load_qa_pipeline():
    model_id = "deependra-2007/distilbert-qa-finetuned"
    return pipeline("question-answering", model=model_id, tokenizer=model_id,device=-1)
def run():
    qa_pipeline = load_qa_pipeline()
    # Streamlit App
    st.title("📚 QA Chatbot with DistilBERT")
    st.write("Ask a question based on the provided context.")
    # Text input for context
    context = st.text_area("Enter context (knowledge passage):", height=200)
    # Text input for question
    question = st.text_input("Enter your question:")
    # On button click, perform QA
    if st.button("Get Answer"):
        if not context.strip() or not question.strip():
            st.warning("Please enter both context and question.")
        else:
            result = qa_pipeline({
                'context': context,
                'question': question
            })
            st.markdown("### ✅ Answer:")
            st.success(result['answer'])
            st.markdown(f"**Confidence Score:** {result['score']:.4f}")
