# Chatbot RAG Prototype

This is a simple chatbot I built to explore how retrieval-based systems and RAG (Retrieval-Augmented Generation) work. It uses Hugging Face embeddings to turn chunks of a text file into vectors, then stores them in a vector database using Chroma. When a user asks a question, the chatbot searches for the most similar chunks and returns them.

## How it works

The chatbot loads a `.txt` file, splits it into smaller parts, and creates embeddings for each chunk. It stores these in Chroma. When a question is asked, it compares the question embedding with the stored ones and shows the most relevant results.

## Technologies used

Python, Hugging Face Transformers, LangChain, ChromaDB

## Why I made this

I wanted to learn more about how RAG systems work and how text retrieval is done using embeddings and vector databases. This is still a beginner project, but it helped me understand the basics and I plan to improve it further.
