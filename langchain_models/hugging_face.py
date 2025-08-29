from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()
import os

api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm=HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text_generation",
    token=api_key
)


model=ChatHuggingFace(llm=llm)
result=model.invoke("what is capital of pakistan")
print(result.content)