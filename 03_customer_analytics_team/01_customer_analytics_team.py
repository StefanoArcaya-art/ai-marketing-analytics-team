

# * LIBRARIES

from langchain_community.vectorstores import Chroma

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from langchain.agents import AgentExecutor, create_vectorstore_agent, create_openapi_agent

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever

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
    
    llm.temperature = temperature
    
    vectorstore_2 = Chroma(
        embedding_function=embedding_function, 
        persist_directory=db_path
    )

    retriever = vectorstore_2.as_retriever()
    
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    
    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \

    {context}"""
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    

    prompt = ChatPromptTemplate.from_template(
        """Answer the question based only on the following context:
        {context}

        Question: {question}
        """
    )
    
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain


product_expert_agent = create_rag_agent(PATH_PRODUCTS_VECTORDB, llm=OPENAI_LLM, temperature=0.7)

result = product_expert_agent.invoke({"input": "Is the 4-Course R-Track Open for Enrollment?", "chat_history": [HumanMessage(content="Is the 4-Course R-Track Open for Enrollment?")]})

result

result['answer']

# * LANGGRAPH

class GraphState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    num_steps: Annotated[Sequence[int], operator.add]
    next: str
    
# Helper function to get last question that the Human asked
def get_last_human_message(msgs):
    # Iterate through the list in reverse order
    for msg in reversed(msgs):
        if isinstance(msg, HumanMessage):
            return msg
    return None
    
    
def supervisor_node(state):
    
    result = supervisor_agent.invoke(state)
    
    print(result)
    
    return {'next': result['next'], 'num_steps': 1}


def product_expert_node(state):
    
    print(state["messages"])
    
    messages = state.get("messages")
    
    last_question = get_last_human_message(messages)
    if last_question:
        last_question = last_question.content
    
    result = product_expert_agent.invoke({"input": last_question, "chat_history": messages})
    
    print(result)
    
    # result = 'No, the 4-Course R-Track is closed for enrollment.'
    
    return {
        "messages": [AIMessage(content=result['answer'], name='Product_Expert')],
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
