from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence
import os

load_dotenv()

groq_api_key= os.getenv("GROQ_API_KEY")

prompt1=PromptTemplate(
    template="write a joke on {topic}",
    input_variables=['topic']
)
prompt2=PromptTemplate(
    template="explain the given joke.\n {joke}",
    input_variables=['joke']
)


model=ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=groq_api_key
)

parser=StrOutputParser()

chain=RunnableSequence(prompt1,model,parser,prompt2,model,parser)

result=chain.invoke({"topic":"AI"})
print(result)
