import sys
from typing import List

import chromadb
from chromadb import Settings, DEFAULT_TENANT, DEFAULT_DATABASE
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_chroma.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.chat_models import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_text_splitters import TextSplitter

llm = ChatOllama(model="llama3")
embeddings = OllamaEmbeddings(model="nomic-embed-text")


class LineByLineTextSplitter(TextSplitter):
    def split_text(self, text: str) -> List[Document]:
        lines = text.split('\n')
        documents = []

        for i, line in enumerate(lines, start=1):
            if line.strip():  # Skip empty lines
                doc = Document(
                    page_content=line.strip(),
                    metadata={"line_number": i}
                )
                documents.append(doc)

        return documents


def generate_embeddings(log_path):
    """
    Generate embeddings from log file with specified name.
    Returns an ephemeral ChromaDB instance.
    """
    print(f'Loading text from log file: {log_path}')
    with open(log_path, 'r') as file:
        text = file.read()
    print(f'Loaded {len(text)} characters from log file')
    text_splitter = LineByLineTextSplitter()
    log_lines = text_splitter.split_text(text)
    print(f'Split {len(log_lines)} lines from log file')
    client = chromadb.EphemeralClient(
        settings=Settings(),
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE,
    )
    batch_size = client.get_max_batch_size()
    batched_documents = [log_lines[i:i + batch_size] for i in range(0, len(log_lines), batch_size)]
    print(f'Creating embeddings with {len(batched_documents)} batches of size {batch_size}')
    db = Chroma(embedding_function=embeddings, client=client)
    for i, batch in enumerate(batched_documents):
        print(f'Processing batch {i}...')
        db.from_documents(embedding=embeddings, documents=batch)
    return db


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Answer questions about the following log file:
            {context}
            """,
        ),
        ("human", "{input}"),
    ]
)

if len(sys.argv) < 2:
    print('Must specify log file name')
    exit(1)

file_name = sys.argv[1]
vector_db: Chroma = generate_embeddings(file_name)
retriever = vector_db.as_retriever(search_kwargs={"k": 10})
question_answer_chain = create_stuff_documents_chain(llm, prompt)
chain = create_retrieval_chain(retriever, question_answer_chain)

while True:
    query = input('Query: ')
    print(f'Processing...')
    response = chain.invoke({"input": query})
    answer = response['answer']
    print('---- Context -----')
    docs: List[Document] = response['context']
    for doc in docs:
        line_num = doc.metadata['line_number']
        content = doc.page_content
        print(f' [{line_num}] {content}')
    print('\n----- Answer -----')
    print(answer)
    print('\n')
