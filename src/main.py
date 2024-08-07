import os
from pathlib import Path
from pprint import pprint

from dotenv import load_dotenv
import torch
from groq import Groq, GroqError
from sentence_transformers import SentenceTransformer

from src.embedding import Embedder
from src.preprocessing import Preprocessing

import warnings
warnings.filterwarnings("ignore")
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def generate_response_with_context(client, query, relevant_docs):
    context = "".join(relevant_docs)
    prompt = f"""Based on the following information:

{context}

Answer the following question: 
{query}
"""
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions based on the given context." # TODO: this could be improved
                },
                {
                    "role": "user",
                    "content": prompt
                }, 

            ],
            model="llama3-groq-70b-8192-tool-use-preview",
            temperature=0.5,
            # max_tokens=1000,
            # stream=True,

        )
    except GroqError as e :
        print(e)
        return "I'm sorry, I couldn't generate a response at the moment."
    
    return chat_completion.choices[0].message.content

def run(load_embedding_path=None, save_path=None) :
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    # preprocessing
    pdf_path = Path(input('File path: '))
    preprocessor = Preprocessing(pdf_path, chunk_size=13)
    data = preprocessor()
    
    # embedding
    embedding_model = SentenceTransformer(
        model_name_or_path="all-mpnet-base-v2", 
        device=device
    )
    emb = Embedder(data, embedding_model)
    if load_embedding_path :
        emb.load_embedding(load_embedding_path)
    else :
        emb.embed()
    
    if save_path :
        emb.save_embedding(Path(save_path))

    # generation
    client = Groq(
        api_key=os.environ.get("GROQ_API_KEY"),
    )

    query = input(": ")
    context = emb.get_related_content(
        query=query,
        k=1
    )
    response = generate_response_with_context(client, query, context)
    pprint(response)

# Load environment variables from .env file
load_dotenv()

run(load_embedding_path="./data/embeddings.pt")