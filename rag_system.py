from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from github_fetcher import fetch_github_data
import numpy as np

# Initialize models
def initialize_models():
    print("Initializing models...")
    model_name = "ibm-granite/granite-34b-code-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    return tokenizer, model, sentence_model

# Global variables for models
tokenizer, model, sentence_model = None, None, None

def retrieve_relevant_docs(query, documents, top_k=3):
    query_embedding = sentence_model.encode([query])
    doc_embeddings = sentence_model.encode(documents)
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [documents[i] for i in top_indices]

def rag_generate(query, documents):
    print("Retrieving relevant documents...")
    relevant_docs = retrieve_relevant_docs(query, documents)
    print(f"Number of relevant documents: {len(relevant_docs)}")

    context = "\n".join(relevant_docs[:3])  # Limit context to first 3 relevant documents
    prompt = f"Context:\n{context}\n\nQuery: {query}\n\nResponse:"
    print("Generated prompt (truncated):", prompt[:100] + "...")

    print("Tokenizing input...")
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    print("Generating response...")
    outputs = model.generate(**inputs, max_length=200)
    print("Decoding response...")
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response

if __name__ == "__main__":
    print("Starting RAG system...")
    try:
        # Initialize models
        tokenizer, model, sentence_model = initialize_models()

        # repo_name = "semaphore-protocol/semaphore"  # Replace with the actual GitHub repository name
        # documents = fetch_github_data(repo_name)
        
        # print(documents)
        
        # query = input("Enter your query: ")
        # print("Query:", query)
        # print(f"Number of documents: {len(documents)}")
        
        # result = rag_generate(query, documents)
        # print("Final result:", result)
    except Exception as e:
        print(f"An error occurred: {str(e)}")