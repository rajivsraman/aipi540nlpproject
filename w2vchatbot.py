import os
import openai
import numpy as np
import chromadb
from gensim.models import Word2Vec
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

class Word2VecChatbot:
    def __init__(self, folder_path):
        """
        Initialize the chatbot with Word2Vec-based document embeddings stored in ChromaDB.
        Uses API key from Streamlit Cloud Secrets.
        """
        self.folder_path = folder_path
        self.api_key = os.getenv("RAJIV_OPENAI_API_KEY")  # Use Streamlit Secrets in app.py
        self.openai_client = openai.OpenAI(api_key=self.api_key)

        # Initialize ChromaDB
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.get_or_create_collection(name="chemistry_knowledge")

        # Load and preprocess documents
        self.docs, self.filenames = self.load_documents()
        self.model = self.train_word2vec()
        self.store_embeddings()

    def load_documents(self):
        """Loads text documents from the specified folder."""
        docs, filenames = [], []
        for file in os.listdir(self.folder_path):
            if file.endswith(".txt"):
                with open(os.path.join(self.folder_path, file), "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    docs.append(content)
                    filenames.append(file.replace(".txt", ""))
        return docs, filenames

    def train_word2vec(self):
        """Trains a Word2Vec model using the text documents."""
        tokenized_docs = [doc.split() for doc in self.docs]
        return Word2Vec(sentences=tokenized_docs, vector_size=100, window=5, min_count=1, workers=4)

    def get_embedding(self, text):
        """Converts text into a Word2Vec embedding by averaging word vectors."""
        words = text.split()
        vectors = [self.model.wv[word] for word in words if word in self.model.wv]
        return np.mean(vectors, axis=0) if vectors else np.zeros(self.model.vector_size)

    def store_embeddings(self):
        """Stores Word2Vec embeddings in ChromaDB."""
        for doc, filename in zip(self.docs, self.filenames):
            vector = self.get_embedding(doc).tolist()
            self.collection.add(
                ids=[filename],
                embeddings=[vector],
                metadatas=[{"source": filename, "content": doc}]
            )

    def retrieve_relevant_docs(self, query, top_n=3):
        """Retrieves the most relevant documents from ChromaDB using its built-in similarity search."""
        query_embedding = self.get_embedding(query).tolist()
        results = self.collection.query(query_embeddings=[query_embedding], n_results=top_n)

        return "\n".join([doc["content"] for doc in results["metadatas"][0]]) if results["metadatas"] else "No relevant documents found."

    def generate_response(self, context, user_query):
        """Generates a response using OpenAI GPT based on retrieved context."""
        prompt = f"Context: {context}\n\nUser query: {user_query}\n\nProvide a concise answer based on the context."
        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a chemistry assistant. ONLY use the provided context to generate responses. If the context does not contain the answer, respond with 'I don't know based on the provided context.' Do NOT use any external knowledge."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content

    def chatbot(self, user_query):
        """Main chatbot function to process user queries."""
        context = self.retrieve_relevant_docs(user_query)
        return self.generate_response(context, user_query)
