import streamlit as st
from streamlit_chat import message
import os

from src.app.utils import generate_response_with_context, prep_docs, create_embedding

st.set_page_config(
    page_title="RAG System",
    page_icon=":mag:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("Configuration")

# PDF file upload
uploaded_file = st.sidebar.file_uploader("Upload PDF file", type="pdf")

# Choosing K : (Relevant Documents)
K = st.sidebar.slider("Number of retrieved relevant documents", min_value=1, max_value=10, value=5)

# LLM model selection
llm_option = st.sidebar.selectbox("Select LLM model", ["OpenAI", "Anthropic", "Groq"])

# API key input
if llm_option == "OpenAI":
    openai_api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")
elif llm_option == "Anthropic":
    anthropic_api_key = st.sidebar.text_input("Enter your Anthropic API key", type="password")
elif llm_option == "Groq":
    groq_api_key = st.sidebar.text_input("Enter your Groq API key", type="password")

# Main content area
st.title("RAG System")

if uploaded_file is not None:
    # Save the PDF file
    pdf_path = os.path.join("data", uploaded_file.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.write(f"PDF file '{uploaded_file.name}' uploaded successfully.")

    # Initialize chat history if not already present
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = [
            {"role": "ai", "content": "Hello there. How can I help you?"}
        ]

    # Chat input
    user_query = st.chat_input("", key="chat_input")

    if user_query:
        # Append user message to chat history
        st.session_state.chat_history.append({"role": "human", "content": user_query})

        # Preprocess and embed the PDF file if not done already
        if "preprocessed_data" not in st.session_state or "embeddings_model" not in st.session_state:
            with st.spinner("Preprocessing data..."):
                st.session_state.preprocessed_data = prep_docs(pdf_path)
            with st.spinner("Creating embeddings..."):
                st.session_state.embeddings_model = create_embedding(st.session_state.preprocessed_data)

        # Get related content
        context = st.session_state.embeddings_model.get_related_content(
            query=user_query,
            k=K
        )
        
        # analysis on context 
        print(context)
        print("#"*20)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        print(f"Token count: {len(tokenizer.encode(context))}")

        # Generate response
        with st.spinner("Generating response..."):
            if llm_option == "OpenAI":
                answer = generate_response_with_context("OpenAI", user_query, context, api_key=openai_api_key)
            elif llm_option == "Anthropic":
                answer = generate_response_with_context("Anthropic", user_query, context, api_key=anthropic_api_key)
            elif llm_option == "Groq":
                answer = generate_response_with_context("Groq", user_query, context, api_key=groq_api_key)

            # Append AI response to chat history
            st.session_state.chat_history.append({"role": "ai", "content": answer})

        # Render updated chat history
        st.subheader("Chat")
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

    # Add a button to clear the chat history
    if st.button("Clear Chat History"):
        st.session_state.chat_history = [{"role": "ai", "content": "Hello there. How can I help you?"}]
        # Clear the chat container
        st.experimental_rerun()  # Rerun the script to clear the container

