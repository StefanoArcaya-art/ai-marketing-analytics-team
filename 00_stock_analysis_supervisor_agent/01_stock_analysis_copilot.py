

# NOTE: requires yfinance to get the SPY data
# NOTE: Requires Tavily API Key for Web Search (add to credentials.yml file)

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

# ['FINISH', 'Researcher', 'Coder']
route_options = ["FINISH"] + subagent_names 

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


# * Research Agent

researcher_agent = create_agent(
    llm, 
    [tavily_tool], 
    "You are a web researcher."
)

# * Code Agent

coder_agent = create_agent(
    llm,
    [python_repl_tool],
    "You may generate safe python code to analyze data and generate charts using Plotly. Please share the specific details of the Python code in your reponse using ```python ``` markdown. Please make sure to use the plotly library.",
)



# * LANGGRAPH

class GraphState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    num_steps : int
    next: str
    
    
def supervisor_node(state):
    
    return supervisor_chain

def research_node(state):
    
    result = researcher_agent.invoke(state)
    
    return {"messages": [HumanMessage(content=result["output"], name="Researcher")]}

def coder_node(state):
    
    result = coder_agent.invoke(state)
    
    return {"messages": [HumanMessage(content=result["output"], name="Coder")]}

  
# def state_printer(state):
#     """print the state"""
#     print("---STATE PRINTER---")
#     print(f"Messages: {state['messages']}")
#     print(f"Formatted Question (SQL): {state['formatted_user_question_sql_only']}")
#     print(f"SQL Query: \n{state['sql_query']}\n")
#     print(f"Data: \n{pd.DataFrame(state['data'])}\n")
#     print(f"Chart or Table: {state['routing_preprocessor_decision']}")
    
#     if state['routing_preprocessor_decision'] == "chart":
#         print(f"Chart Code: \n{pprint(state['chart_plotly_code'])}")
#         print(f"Chart Error: {state['chart_plotly_error']}")
    
#     print(f"Num Steps: {state['num_steps']}")

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

graph = workflow.compile()

Image(graph.get_graph().draw_mermaid_png())



# * TESTING THE STOCK ANALYSIS COPILOT

      
result_3 = graph.invoke(
    input = {"messages": [HumanMessage(content="Find the historical prices of SPY for the last 5 years from Yahoo Finance (feel free to use the yfinance library, which is installed). Plot a daily line chart of the SPY value over time from the historical prices using python and the plotly library. Add a 50-day and 200-day simple moving average.")]},
    config = {"recursion_limit": 10},
)

pprint(dict(result_3), width=40, compact=True)

for message in result_3['messages']:
    if message.name:
        print(f"Name: {message.name}")
    print(f"Content: {message.content}")
    print("---")
    print()
    
  
result_4 = graph.invoke(
    input = {"messages": [HumanMessage(content="Find the historical prices of NVDA and VIX for the last 5 years from Yahoo Finance (feel free to use the yfinance library, which is installed). Plot a daily line chart of the value over time from the historical prices using python and the plotly library. Organize the plots by using 1 column by 2 row subplots so that the dates line up and VIX is the first plot and NVDA is below. Make sure the end date used is '2024-07-24'. Add a dateslider.")]},
    config = {"recursion_limit": 10},
)  

for message in result_4['messages']:
    if message.name:
        print(f"Name: {message.name}")
    print(f"Content: {message.content}")
    print("---")
    print()
