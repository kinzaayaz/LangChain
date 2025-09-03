from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from dotenv import load_dotenv
import os

load_dotenv()

api_key= os.getenv("GROQ_API_KEY")

model=ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=api_key
)

loader=PyPDFLoader("Introduction to Compiler Construction.pdf")
docs=loader.load()

splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0
)
result= splitter.split_documents(docs)

print(result[1].page_content)
print(result[1].metadata)
