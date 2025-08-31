from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

api_key=os.getenv("GROQ_API_KEY")

model=ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=api_key
)

prompt_template=PromptTemplate(
    template="write a five line summary on {topic}",
    input_variables=['topic']
)

parser=StrOutputParser()

chain= prompt_template | model | parser

result = chain.invoke({"topic":"AI"})
print(result)

# chain.get_graph().print_ascii()