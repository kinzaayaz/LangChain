from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence,RunnableParallel,RunnablePassthrough
import os

load_dotenv()

groq_api_key= os.getenv("GROQ_API_KEY")

prompt1=PromptTemplate(
    template="write a joke on {topic}",
    input_variables=['topic']
)
prompt2=PromptTemplate(
    template="explain the given joke in 3 lines.\n {joke}",
    input_variables=['joke']
)


model=ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=groq_api_key
)

parser=StrOutputParser()

joke_gen_chain=RunnableSequence(prompt1,model,parser)

parallel_chain=RunnableParallel({
    "joke":RunnablePassthrough(),
    "explanation":RunnableSequence(prompt2,model,parser)
})

final_chain=RunnableSequence(joke_gen_chain,parallel_chain)

result=final_chain.invoke({"topic":"AI"})
print(result['joke'])
print(result['explanation'])
