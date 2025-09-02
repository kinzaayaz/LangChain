from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough,RunnableLambda,RunnableBranch
import os

load_dotenv()

groq_api_key= os.getenv("GROQ_API_KEY")

prompt=PromptTemplate(
    template="write a detailed report on {topic}",
    input_variables=['topic']
)
prompt1=PromptTemplate(
    template="generate a short summary on following text\n {text}",
    input_variables=['text']
)

model=ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=groq_api_key
)

parser=StrOutputParser()

report_chain=RunnableSequence(prompt,model,parser)

brach_chain=RunnableBranch(
    (lambda x: len(x.split())>500 , RunnableSequence(prompt1,model,parser)),
    RunnablePassthrough()
)

chain=RunnableSequence(report_chain,brach_chain)
result=chain.invoke({"topic":"flood vs pollution"})
print(result)