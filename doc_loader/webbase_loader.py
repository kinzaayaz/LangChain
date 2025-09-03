from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()
api_key= os.getenv("GROQ_API_KEY")

url="https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.base.Runnable.html"
loader= WebBaseLoader(url)
doc=loader.load()


prompt=PromptTemplate(
    template="answer the following {question} from the following text.\n {text}",
    input_variables=['question','text']
)

model=ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=api_key
)

parser = StrOutputParser()

chain=prompt|model|parser

print(chain.invoke({"question":"what are the types of runnables?","text":doc[0].page_content}))
