from pathlib import Path
from pprint import pprint

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

class Embedder:
    def __init__(self, data: pd.DataFrame, embedding_model: SentenceTransformer) -> None:
        self.data = data
        self.embedding_model = embedding_model
        self.embeddings = None
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    def embed(self) -> None:
        text_chunk_embeddings = self.data.apply(
            func=lambda row: self.embedding_model.encode(
                row['chunked'],
                batch_size=64, 
            ),
        axis=1
        )
        # convert to tensor
        text_chunk_embeddings = text_chunk_embeddings.apply(
            func=lambda embedding: torch.tensor(embedding)
        )
        # convert the embeddings into a matrix
        self.embeddings = torch.stack(text_chunk_embeddings.tolist()).to(self.device)
    
    def load_embedding(self, path: Path) -> None:
        self.embeddings = torch.load(path)
    
    def save_embedding(self, path: Path) -> None:
        torch.save(self.embeddings, path)

    def get_score(self, query, k):
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=True)
        dot_score = util.dot_score(a=query_embedding, b=self.embeddings)
        score_result = torch.topk(
            input=dot_score,
            k=k,
            dim=1
        )
        return score_result
    
    def print_related_content(self, query, k=5):
        score_result = self.get_score(query, k)
        for value, index in list(zip(score_result[0].ravel(), score_result[1].ravel())) :
            index = int(index)
            page_number = self.data.iloc[index]['page_number']
            print(f"Score: {value}")
            print(f"Index: {index}")
            print(f"Page: {page_number}")
            pprint(self.data.iloc[index]['chunked'])
            print()

    def get_related_content(self, query, k=5):
        score_result = self.get_score(query, k)
        for value, index in list(zip(score_result[0].ravel(), score_result[1].ravel())) :
            index = int(index)
            return self.data.iloc[index]['chunked']
        


if __name__ == "__main__":
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    # preprocessing
    from src.preprocessing import Preprocessing
    pdf_path = Path(input('File path: '))
    preprocessor = Preprocessing(pdf_path, chunk_size=13)
    data = preprocessor()
    
    embedding_model = SentenceTransformer(
        model_name_or_path="all-mpnet-base-v2", 
        device=device
    )


    emb = Embedder(data, embedding_model)
    emb.embed(data=data)
    # emb.load_embedding('./data/embeddings.pt')
    # emb.save_embedding(Path('./data/embeddings.pt'))
    emb.print_related_content(
        input("Ask: ")
    )
    