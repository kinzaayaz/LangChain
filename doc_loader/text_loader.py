from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()
api_key= os.getenv("GROQ_API_KEY")


prompt=PromptTemplate(
    template="write a short summary on followinf poem.\n {poem}",
    input_variables=['poem']
)

model=ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=api_key
)

parser = StrOutputParser()

loader = TextLoader("peace.txt")
doc=loader.load()

chain = prompt | model | parser
result=chain.invoke({"poem":doc[0].page_content})
print(type(doc))

print(len(doc))

print(doc[0].page_content)

print(doc[0].metadata)
print(result)
