
from pathlib import Path

import fitz  # pymupdf
import numpy as np
import pandas as pd
from spacy.lang.en import English


class Preprocessing:
    def __init__(self, file_path: Path, chunk_size: int) -> None:
        self.chunk_size = chunk_size
        self.data = self.prepare_data(file_path)
        self.split_to_pages()
        self.chunk_data()
        self.split_chunks()

    def __call__(self):
        return self.data
    
    def prepare_data(self, file_path): # TODO this procedure can be optimized later
        with fitz.open(file_path) as doc:
            page_content = {}
            for page_number, page in enumerate(doc) :
                text = page.get_text().replace('\n', ' ')
                page_content[page_number+1] = text

            data = pd.DataFrame(columns=['page_number', 'text'])
            data['page_number'] = list(page_content.keys())
            data['text'] = list(page_content.values())
            del page_content

            return data

    def split_to_pages(self):
        nlp = English()
        nlp.add_pipe("sentencizer")

        self.data['sentences'] = self.data.apply(
            func=lambda row: list(nlp(row['text']).sents),
            axis=1,
            # result_type='expand'
        )
        # make sure all the sentences are str() (if you don't do this , the type will be )
        self.data['sentences'] = self.data['sentences'].map(
            lambda sentences: list(map(lambda s: s.text, sentences))
        )

    def chunk_data(self):
        def calculate_chunk_size():
            data['sentences']

        def get_chunk_points(chunk_size, array_length):
            return list(
                range(chunk_size, array_length + 1, chunk_size)
            )
        
        self.data['chunked'] = self.data.apply(
            func=lambda row: np.split(
                row['sentences'],
                get_chunk_points(self.chunk_size, len(row['sentences']))
            ),
            axis=1
        )

    def split_chunks(self):
        chunked_data = self.data.explode('chunked').reset_index()
        chunked_data = chunked_data[['page_number', 'chunked']]

        # converting the chunked list into one string
        chunked_data['chunked'] = chunked_data['chunked'].apply(
            func=lambda chunk: ' '.join(chunk)
        )

        self.data = chunked_data

    def save_data(self, path: Path):
        self.data.to_csv(path, index=False)

if __name__ == '__main__':
    pdf_path = Path('./Early Iran History.pdf')
    preprocessor = Preprocessing(pdf_path, chunk_size=13)
    # preprocessor.save_data(Path('./data/chunked_data.csv'))
    data = preprocessor()
    