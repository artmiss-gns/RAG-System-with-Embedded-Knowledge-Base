
from pathlib import Path

import anthropic
import openai
import torch
from groq import Groq, GroqError
from sentence_transformers import SentenceTransformer
from streamlit_chat import message

from src.embedding import Embedder
from src.preprocessing import Preprocessing

device = 'mps' if torch.backends.mps.is_available() else 'cpu'

def generate_response_with_context(llm_option, query, relevant_docs, api_key):
    context = "".join(relevant_docs)
    prompt = f"""Based on the following information:

{context}

Answer the following question: 
{query}
"""

    if llm_option == "OpenAI":
        try:
            openai.api_key = api_key
            response = openai.ChatCompletion.create(
                model="oai-gpt-3.5-turbo-instruct",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on the given context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1024,
                n=1,
                stop=None,
                temperature=0.5,
            )
            return response.choices[0].message.content.strip()
        except openai.error.OpenAIError as e:
            print(e)
            return e

    elif llm_option == "Anthropic":
        try:
            anthropic.api_key = api_key
            chat_completion = anthropic.ChatCompletion.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that answers questions based on the given context."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    },
                ],
                model="llama3-groq-70b-8192-tool-use-preview",
                temperature=0.5,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            print(e)
            return e

    elif llm_option == "Groq":
        try:
            client = Groq(api_key=api_key)
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that answers questions based on the given context."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    },
                ],
                model="llama3-groq-70b-8192-tool-use-preview",
                temperature=0.5,
                max_tokens=1024,
                # stream=True,
            )
            return chat_completion.choices[0].message.content
        except GroqError as e:
            print(e)
            return e

def prep_docs(pdf_path, chunk_size=13):
    print("Preprocessing...")
    preprocessor = Preprocessing(pdf_path, chunk_size=chunk_size)
    data = preprocessor()
    
    return data

def create_embedding(data) -> Embedder:
    embedding_model = SentenceTransformer(
        model_name_or_path="all-mpnet-base-v2", 
        device=device
    )
    emb = Embedder(data, embedding_model)
    print("Embedding...")
    emb.embed()

    return emb