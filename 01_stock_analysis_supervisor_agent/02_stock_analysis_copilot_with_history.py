# BUSINESS SCIENCE UNIVERSITY
# PYTHON FOR GENERATIVE AI COURSE
# MULTI-AGENTS (AGENTIAL SUPERVISION)
# ***

# Goal: Add chat memory so workers know what's been done previously (Checkpointing with SqliteSaver)

# NOTE: requires yfinance to get the SPY data
# NOTE: Requires Tavily API Key for Web Search (add to credentials.yml file)

# * LIBRARIES

# * NEW: Used for conversation memory
from langgraph.checkpoint.sqlite import SqliteSaver

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.tools import PythonREPLTool

import operator
import functools
import os
import yaml

import yfinance 

from typing import Annotated, Sequence, TypedDict

from pprint import pprint
from IPython.display import Image

# * MEMORY SETUP
# - We can have either a database on server or a JIT "in-memory" database

memory = SqliteSaver.from_conn_string(":memory:")

# * LLM SELECTION

MODEL = "gpt-4o-mini"
# MODEL = "gpt-3.5-turbo"
# MODEL = "gpt-4o"

# * API KEYS

os.environ["OPENAI_API_KEY"] = yaml.safe_load(open('../credentials.yml'))['openai']

os.environ["TAVILY_API_KEY"] = yaml.safe_load(open("../credentials.yml"))['tavily']

# * Create tools

tavily_tool = TavilySearchResults(max_results=5)

python_repl_tool = PythonREPLTool()

# * Create Agent Supervisor
#   - Supervisor has 1 role: Pick which team member to send to (or if finished)

subagent_names = ["Researcher", "Coder"]

system_prompt = (
    "You are a supervisor tasked with managing a conversation between the following workers:  {subagent_names}. Given the following user request, respond with the worker to act next. Each worker will perform a task and respond with their results and status. When finished, respond with FINISH."
)

# Our team supervisor is an LLM node. It just picks the next agent to process and decides when the work is completed

# ['FINISH', 'Researcher', 'Coder']
route_options = ["FINISH"] + subagent_names 
route_options

# Using openai function calling can make output parsing easier for us
#  References: 
#   https://platform.openai.com/docs/guides/function-calling 
#   https://cookbook.openai.com/examples/how_to_call_functions_with_chat_models

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

pprint(dict(prompt))

llm = ChatOpenAI(model=MODEL)

supervisor_chain = (
    prompt
    | llm.bind_functions(functions=[function_def], function_call="route")
    | JsonOutputFunctionsParser()
)

supervisor_chain

# * SUBAGENTS

# * Helper function

def create_agent_with_tools(llm: ChatOpenAI, tools: list, system_prompt: str):
    # Each worker node will be given a name and some tools.
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor


# * Researcher Agent

researcher_agent = create_agent_with_tools(
    llm, 
    [tavily_tool], 
    "You are a web researcher."
)

researcher_agent

# * Coder Agent

coder_agent = create_agent_with_tools(
    llm,
    [python_repl_tool],
    "You may generate safe python code to analyze data and generate charts using Plotly. Please share the specific details of the Python code in your reponse using ```python ``` markdown. Please make sure to use the plotly library.",
)

coder_agent

# * LANGGRAPH

#   - NEW Skill: Annotated Sequences
class GraphState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    num_steps: Annotated[Sequence[int], operator.add]
    next: str
    
    
def supervisor_node(state):
    
    result = supervisor_chain.invoke(state)
    
    print(result)
    
    return {'next': result['next'], 'num_steps': 1}

def research_node(state):
    
    result = researcher_agent.invoke(state)
    
    return {
        "messages": [AIMessage(content=result["output"], name="Researcher")],
        'num_steps': 1
    }

def coder_node(state):
    
    result = coder_agent.invoke(state)
    
    return {
        "messages": [AIMessage(content=result["output"], name="Coder")],
        'num_steps': 1
    }


# * WORKFLOW DAG

workflow = StateGraph(GraphState)

workflow.add_node("Researcher", research_node)
workflow.add_node("Coder", coder_node)
workflow.add_node("supervisor", supervisor_node)

for member in subagent_names:
    workflow.add_edge(member, "supervisor")
    
conditional_map = {'Researcher': 'Researcher', 'Coder': 'Coder', 'FINISH': END}

workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)

workflow.set_entry_point("supervisor")

# * NEW: ADD CHECKPOINTER MEMORY
app = workflow.compile(checkpointer=memory)

Image(app.get_graph().draw_mermaid_png())



# * TESTING THE STOCK ANALYSIS COPILOT

      
result_3 = app.invoke(
    input = {"messages": [HumanMessage(content="Find the historical prices of SPY for the last 5 years from Yahoo Finance (feel free to use the yfinance library, which is installed). Plot a daily line chart of the SPY value over time from the historical prices using python and the plotly library. Add a 50-day and 200-day simple moving average. Make sure the end date used is '2024-07-24'. Add a dateslider.")]},
    
    # * NEW: Add thread_id
    config = {"recursion_limit": 10, "configurable": {"thread_id": "1"}},
)

result_3

for message in result_3['messages']:
    if message.name:
        print(f"Name: {message.name}")
    print(f"Content: {message.content}")
    print("---")
    print()
    
app.invoke(
    input = {"messages": [HumanMessage(content="Reproduce the last plot")]},
    
    # * NEW: Add thread_id
    config = {
        "recursion_limit": 10, 
        "configurable": {"thread_id": "1"}
    },
)
  
result_4 = app.invoke(
    input = {"messages": [HumanMessage(content="Find the historical prices of NVDA and VIX for the last 1 year from Yahoo Finance (feel free to use the yfinance library, which is installed). Plot a daily line chart of the value over time from the historical prices using python and the plotly library. Organize the plots by using 1 column by 2 row subplots so that the dates line up and VIX is the first plot and NVDA is below. Make sure the end date used is '2024-07-24'")]},
    
    config = {
        "recursion_limit": 10, 
        "configurable": {"thread_id": "1"}
    },
)  

result_4

for message in result_4['messages']:
    if message.name:
        print(f"Name: {message.name}")
    print(f"Content: {message.content}")
    print("---")
    print()


# * NEW: Working with messages from the database (checkpointer)

config = {"configurable": {"thread_id": "1"}}

app.get_state(config).values

messages = app.get_state(config).values["messages"]
messages

