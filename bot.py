from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
import os

loader = TextLoader('your_document.txt')   
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")

#print("Documents loaded, split, embedded, and stored in ChromaDB!")

db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

print("\nSetup complete! You can now ask questions about your document.\n")

while True:
    query = input("\nAsk a question (or type 'exit' to quit): ")
    
    if len(query.split()) <= 2:
        query = "Tell me about " + query

    if query.lower() == "exit":
        print("Goodbye!")
        break

   
    results_and_scores = db.similarity_search_with_score(query, k=3)
   
    MIN_SIMILARITY_THRESHOLD = 0.10   

    good_results = []
    for doc, score in results_and_scores:
        similarity = 1 - score   
        if similarity >= MIN_SIMILARITY_THRESHOLD:
            good_results.append(doc.page_content)

    if good_results:
        unique_chunks = list(set(good_results))
        combined_text = "\n".join(unique_chunks)
        print("\nAnswer:\n", combined_text)

    else:
        print("\nSorry, I don't have enough information to answer that.")
