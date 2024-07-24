

# will recommend to use the yfinance library

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import BaseMessage, HumanMessage
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


MODEL = "gpt-4o-mini"
# MODEL = "gpt-3.5-turbo"
# MODEL = "gpt-4o"

# * API KEYS

os.environ["OPENAI_API_KEY"] = yaml.safe_load(open('../credentials.yml'))['openai']

os.environ["TAVILY_API_KEY"] = yaml.safe_load(open("../credentials.yml"))['tavily']

# * Create tools

tavily_tool = TavilySearchResults(max_results=5)

python_repl_tool = PythonREPLTool()

# * Helper Utilities

def create_agent(llm: ChatOpenAI, tools: list, system_prompt: str):
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

def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}


# * Create Agent Supervisor

subagent_names = ["Researcher", "Coder"]

system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    " following workers:  {subagent_names}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
)

# Our team supervisor is an LLM node. It just picks the next agent to process and decides when the work is completed

route_options = ["FINISH"] + subagent_names # ['FINISH', 'Researcher', 'Coder']

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

# * Research Agent

research_agent = create_agent(llm, [tavily_tool], "You are a web researcher.")

# * Code Agent

code_agent = create_agent(
    llm,
    [python_repl_tool],
    "You may generate safe python code to analyze data and generate charts using matplotlib.",
)



# * LANGGRAPH

class GraphState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str
    
research_node = functools.partial(agent_node, agent=research_agent, name="Researcher")

code_node = functools.partial(agent_node, agent=code_agent, name="Coder")



workflow = StateGraph(GraphState)

workflow.add_node("Researcher", research_node)
workflow.add_node("Coder", code_node)
workflow.add_node("supervisor", supervisor_chain)

for member in subagent_names:
    workflow.add_edge(member, "supervisor")
    
conditional_map = {'Researcher': 'Researcher', 'Coder': 'Coder'}
conditional_map["FINISH"] = END

workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)

workflow.set_entry_point("supervisor")

graph = workflow.compile()

Image(graph.get_graph().draw_mermaid_png())



# * TESTING THE STOCK ANALYSIS COPILOT

      
result_3 = graph.invoke(
    input = {"messages": [HumanMessage(content="Find the historical prices of SPY for the last 5 years from Yahoo Finance (feel free to use the yfinance library, which is installed). Plot a daily line chart of the SPY value over time from the historical prices using python.")]},
    config = {"recursion_limit": 40},
)

pprint(dict(result_3), width=40, compact=True)

for message in result_3['messages']:
    if message.name:
        print(f"Name: {message.name}")
    print(f"Content: {message.content}")
    print("---")
    print()
