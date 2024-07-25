

# * LIBRARIES

from langchain_community.vectorstores import Chroma

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from langchain.agents import AgentExecutor, create_vectorstore_agent, create_openapi_agent

from langgraph.graph import StateGraph, END

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from typing import Annotated, Sequence, TypedDict

import operator

import pandas as pd
import yaml
import os

from pprint import pprint
from IPython.display import Image

# * DATABASE SETUP

PATH_PRODUCTS_VECTORDB = "data/data-rag-product-information/products_clean.db"

PATH_TRANSACTIONS_DATABASE = ""


# * AI SETUP

os.environ["OPENAI_API_KEY"] = yaml.safe_load(open('../credentials.yml'))['openai']

OPENAI_LLM = ChatOpenAI(
    model = "gpt-4o-mini"
)

# *** SUPERVISOR AGENT ***

subagent_names = ["Product_Expert", "Customer_Transactions_Expert", "Documenter"]

def create_supervisor_agent(subagent_names: list, llm, temperature=0):

    system_prompt = (
        "You are a supervisor tasked with managing a conversation between the following workers:  {subagent_names}. Given the following user request, respond with the worker to act next. Each worker will perform a task and respond with their results and status. When finished, respond with FINISH."
    )

    route_options = ["FINISH"] + subagent_names 

    function_def = {
        "name": "route",
        "description": "Select the next role.",
        "parameters": {
            "title": "route_schema",
            "type": "object",
            "properties": {
                "next": {
                    "title": "Next",
                    "anyOf": [
                        {"enum": route_options},
                    ],
                }
            },
            "required": ["next"],
        },
    }

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                "Given the conversation above, who should act next?"
                " Or should we FINISH? Select one of: {route_options}",
            ),
        ]
    ).partial(route_options=str(route_options), subagent_names=", ".join(subagent_names))
    
    llm.temperature = temperature
    
    supervisor_chain = (
        prompt
        | llm.bind_functions(functions=[function_def], function_call="route")
        | JsonOutputFunctionsParser()
    )
    
    return supervisor_chain
    

supervisor_agent = create_supervisor_agent(subagent_names=subagent_names, llm=OPENAI_LLM, temperature=0.7)

result = supervisor_agent.invoke({"messages": [HumanMessage(content="Is the 4-Course R-Track Open for Enrollment?")]})

result

# *** PRODUCTS EXPERT RAG AGENT ****

def create_rag_agent(db_path, llm, temperature = 0):
    
    embedding_function = OpenAIEmbeddings(
        model='text-embedding-ada-002',
    )
    
    model = llm
    
    model.temperature = temperature
    
    vectorstore_2 = Chroma(
        embedding_function=embedding_function, 
        persist_directory=db_path
    )

    retriever_2 = vectorstore_2.as_retriever()

    prompt = ChatPromptTemplate.from_template(
        """Answer the question based only on the following context:
        {context}

        Question: {question}
        """
    )

    rag_chain = (
        {"context": retriever_2, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    
    return rag_chain


product_expert_agent = create_rag_agent(PATH_PRODUCTS_VECTORDB, llm=OPENAI_LLM, temperature=0.7)

result = product_expert_agent.invoke(input="Is the 4-Course R-Track Open for Enrollment?")
result



# * LANGGRAPH

class GraphState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    num_steps: Annotated[Sequence[int], operator.add]
    next: str
    
    
def supervisor_node(state):
    
    result = supervisor_agent.invoke(state)
    
    print(result)
    
    return {'next': result['next'], 'num_steps': 1}


def product_expert_node(state):
    
    print(state["messages"])
    
    result = product_expert_agent.invoke(state.get("messages")[-1].content)
    
    # result = 'No, the 4-Course R-Track is closed for enrollment.'
    
    return {
        "messages": [AIMessage(content=result, name='Product_Expert')],
        'num_steps': 1
    }
    
def customer_transactions_expert_node(state):
    
    result = "TEST"
    
    return {
        "messages": [AIMessage(content=result, name='Product_Expert')],
        'num_steps': 1
    }


def documenter_node(state):
    
    result = "TEST"
    
    return {
        "messages": [AIMessage(content=result, name='Product_Expert')],
        'num_steps': 1
    }

# * WORKFLOW DAG

workflow = StateGraph(GraphState)

workflow.add_node("supervisor", supervisor_node)
workflow.add_node("Product_Expert", product_expert_node)
workflow.add_node("Customer_Transactions_Expert", customer_transactions_expert_node)
workflow.add_node("Documenter", documenter_node)

for member in subagent_names:
    workflow.add_edge(member, "supervisor")

conditional_map = {
    'Product_Expert': 'Product_Expert', 
    'Customer_Transactions_Expert': 'Customer_Transactions_Expert', 
    'Documenter':'Documenter',
    'FINISH': END
}
workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)

workflow.set_entry_point("supervisor")

app = workflow.compile()

Image(app.get_graph().draw_mermaid_png())

# * TESTING THE BUSINESS INTELLIGENCE TEAM COPILOT


result = app.invoke(
    input = {"messages": [HumanMessage(content="Is the 4-Course R-Track Open for Enrollment?")]},
    
    # * NEW: Add thread_id
    config = {"recursion_limit": 4},
)

result
