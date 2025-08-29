from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

doc=[
    "The cat is sleeping on the sofa.",
    "A dog is barking loudly outside.",
    "I love to relax on the couch with my pet cat.",
    "Today I am learning about artificial intelligence and machine learning."
]

query="i am learning artifical intelligence"

doc_embeddings=embedding.embed_documents(doc)
query_embeddings=embedding.embed_query(query)
similarities=cosine_similarity([query_embeddings],doc_embeddings)[0]
index,score = max(list(enumerate(similarities)),key=lambda x:x[1])
print(query)
print(doc[index])
print("similarity score is: ",score)

