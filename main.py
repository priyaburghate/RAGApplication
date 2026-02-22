import os
from dotenv import load_dotenv

from loaders.pdf_loader import load_pdf
from splitters.text_splitter import split_documents
from embeddings.openai_embeddings import get_embeddings
from vectorstore.chroma_store import create_chroma_store, load_chroma_store
from retriever.retriever import get_retriever
from chains.rag_chain import build_rag_chain

load_dotenv()

PDF_PATH = "data/employee_handbook.pdf"
PERSIST_DIR = "chroma_db"


class RAGService:
    def __init__(self):
        embeddings = get_embeddings()

        if not os.path.exists(PERSIST_DIR):
            print("Creating new chroma DB...")
            documents = load_pdf(PDF_PATH)
            chunks = split_documents(documents)
            vectodb = create_chroma_store(
                chunks, embeddings, persist_dir=PERSIST_DIR
            )
        else:
            print("Loading existing chroma DB...")
            vectodb = load_chroma_store(
                embeddings, persist_dir=PERSIST_DIR
            )

        retriever = get_retriever(vectodb)
        self.rag_chain = build_rag_chain(retriever)

    def ask(self, query: str) -> str:
        return self.rag_chain.invoke(query)


def main():
    rag = RAGService()

    print("Employee Handbook RAG Application")
    print("Type 'exit' to quit.")

    while True:
        query = input("Ask a question: ")
        if query.lower() == "exit":
            break

        response = rag.ask(query)
        print("Answer:", response)


if __name__ == "__main__":
    main()
