from langchain_huggingface import HuggingFaceEmbeddings

embeddings=HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

query="i am an artifical intelligence robot."

result=embeddings.embed_query(query)
print(result)
print("Embedding length:", len(result))
print("First 10 values:", result[:10])
