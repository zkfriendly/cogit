from langchain.llms import Ollama
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from github_fetcher import fetch_github_data
import numpy as np

# Initialize models
def initialize_models():
    print("Initializing models...")
    llm = Ollama(model="llama3.1")
    embeddings = OllamaEmbeddings(model="llama3.1")
    return llm, embeddings

# Global variables for models
llm, embeddings = None, None

def create_vectorstore(documents):
    return FAISS.from_texts(documents, embeddings)

def rag_generate(query, vectorstore):
    print("Retrieving relevant documents and generating response...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
    )
    response = qa_chain.run(query)
    return response

if __name__ == "__main__":
    print("\n" + "="*50)
    print("Starting RAG system...")
    print("="*50 + "\n")
    try:
        # Initialize models
        llm, embeddings = initialize_models()
        print("\n" + "-"*50)
        print("Models initialized successfully.")
        print("-"*50 + "\n")

        repo_name = "semaphore-protocol/semaphore"  # Replace with the actual GitHub repository name
        documents = fetch_github_data(repo_name)
        vectorstore = create_vectorstore(documents)
        print("\n" + "-"*50)
        print("Documents fetched and vectorstore created successfully.")
        print(f"Number of documents: {len(documents)}")
        print("-"*50 + "\n")
        
        while True:
            query = input("Enter your query (or 'quit' to exit): ")
            if query.lower() == 'quit':
                print("\n" + "="*50)
                print("Exiting the RAG system. Goodbye!")
                print("="*50 + "\n")
                break
            
            print("\n" + "*"*50)
            print("Query:", query)
            print("*"*50)
            result = rag_generate(query, vectorstore)
            print("\n" + "*"*50)
            print("Response:")
            print(result)
            print("*"*50 + "\n")
    except Exception as e:
        print("\n" + "!"*50)
        print(f"An error occurred: {str(e)}")
        print("!"*50 + "\n")