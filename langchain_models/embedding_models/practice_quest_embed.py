from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
documents=[
    "I love programming in Python.",
    "Python is my favorite language for coding."
]
doc_embeddings=embeddings.embed_documents(documents)
print(cosine_similarity(doc_embeddings))