from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

api_key=os.getenv("GROQ_API_KEY")

prompt1=PromptTemplate(
    template="generate a detailed summary on {topic}",
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template="generate a five line summary on given text./n {text}",
    input_variables=['text']
)

model=ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=api_key
)

parser=StrOutputParser()

chain=prompt1 | model | parser |prompt2 | model | parser

result=chain.invoke({"topic":"AI"})
print(result)
# chain.get_graph().print_ascii()