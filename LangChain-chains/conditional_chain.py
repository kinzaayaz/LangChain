from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain.schema.runnable import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

model1 = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=api_key
)

parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(
        description="Give the sentiment of the feedback"
    )

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template=(
    "Classify the sentiment of the following feedback text into positive or negative:\n{feedback}\n{format_instruction}"),
    input_variables=["feedback"],
    partial_variables={"format_instruction": parser2.get_format_instructions()},
)

classifier_chain = prompt1 | model1 | parser2

prompt2 = PromptTemplate(
    template="Write an appropriate response to this positive feedback:\n{feedback}",
    input_variables=["feedback"],
)

prompt3 = PromptTemplate(
    template="Write an appropriate response to this negative feedback:\n{feedback}",
    input_variables=["feedback"],
)

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "positive", prompt2 | model1 | parser),
    (lambda x: x.sentiment == "negative", prompt3 | model1 | parser),
    RunnableLambda(lambda x: "could not find sentiment"),
)

chain = classifier_chain | branch_chain

print(chain.invoke({"feedback": "This is a beautiful phone"}))
chain.get_graph().print_ascii()
