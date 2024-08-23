from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Initialize LLM
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Initialize sentence transformer
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

def retrieve_relevant_docs(query, documents, top_k=3):
    query_embedding = sentence_model.encode([query])
    doc_embeddings = sentence_model.encode(documents)
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [documents[i] for i in top_indices]

def rag_generate(query, documents):
    print("Retrieving relevant documents...")
    relevant_docs = retrieve_relevant_docs(query, documents)
    print("Relevant documents:", relevant_docs)

    context = "\n".join(relevant_docs)
    prompt = f"Context:\n{context}\n\nQuery: {query}\n\nResponse:"
    print("Generated prompt:", prompt)

    print("Tokenizing input...")
    inputs = tokenizer(prompt, return_tensors="pt")
    print("Generating response...")
    outputs = model.generate(**inputs, max_length=200)
    print("Decoding response...")
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response

# Ensure the script is being run as the main program
if __name__ == "__main__":
    print("Starting RAG system...")
    try:
        documents = [
            "Python is a high-level programming language.",
            "JavaScript is commonly used for web development.",
            "Machine learning involves training models on data."
        ]

        query = "What is Python used for?"
        print("Query:", query)
        print("Documents:", documents)

        result = rag_generate(query, documents)
        print("Final result:", result)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
