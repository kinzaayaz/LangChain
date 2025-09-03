from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader

loader=PyPDFLoader("peace.pdf")
doc=loader.load()
print(doc[0].page_content)
print(doc[0].metadata)