

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
from typing import Annotated, Sequence, TypedDict
import functools
import os
import yaml
from pprint import pprint
from IPython.display import Image



MODEL = "gpt-3.5-turbo"
# MODEL = "gpt-4o"

# * API KEYS

os.environ["OPENAI_API_KEY"] = yaml.safe_load(open('../credentials.yml'))['openai']

os.environ["TAVILY_API_KEY"] = yaml.safe_load(open("../credentials.yml"))['tavily']

# * Create tools

tavily_tool = TavilySearchResults(max_results=5)

# This executes code locally, which can be unsafe
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

members = ["Researcher", "Coder"]

system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    " following workers:  {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
)

# Our team supervisor is an LLM node. It just picks the next agent to process
# and decides when the work is completed
options = ["FINISH"] + members

# Using openai function calling can make output parsing easier for us
function_def = {
    "name": "route",
    "description": "Select the next role.",
    "parameters": {
        "title": "routeSchema",
        "type": "object",
        "properties": {
            "next": {
                "title": "Next",
                "anyOf": [
                    {"enum": options},
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
            " Or should we FINISH? Select one of: {options}",
        ),
    ]
).partial(options=str(options), members=", ".join(members))

llm = ChatOpenAI(model=MODEL)

supervisor_chain = (
    prompt
    | llm.bind_functions(functions=[function_def], function_call="route")
    | JsonOutputFunctionsParser()
)

# * Research and Code Agents

research_agent = create_agent(llm, [tavily_tool], "You are a web researcher.")

research_node = functools.partial(agent_node, agent=research_agent, name="Researcher")

# NOTE: THIS PERFORMS ARBITRARY CODE EXECUTION. PROCEED WITH CAUTION
code_agent = create_agent(
    llm,
    [python_repl_tool],
    "You may generate safe python code to analyze data and generate charts using matplotlib.",
)

code_node = functools.partial(agent_node, agent=code_agent, name="Coder")

# * Construct Graph

# The agent state is the input to each node in the graph
class AgentState(TypedDict):
    # The annotation tells the graph that new messages will always
    # be added to the current states
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # The 'next' field indicates where to route to next
    next: str

workflow = StateGraph(AgentState)

workflow.add_node("Researcher", research_node)
workflow.add_node("Coder", code_node)
workflow.add_node("supervisor", supervisor_chain)

for member in members:
    # We want our workers to ALWAYS "report back" to the supervisor when done
    workflow.add_edge(member, "supervisor")
    
# The supervisor populates the "next" field in the graph state
# which routes to a node or finishes
conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END

workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)

# Finally, add entrypoint
workflow.set_entry_point("supervisor")

graph = workflow.compile()

Image(graph.get_graph().draw_mermaid_png())

# * Invoke the team

for s in graph.stream(
    {
        "messages": [
            HumanMessage(content="Code hello world and print it to the terminal")
        ]
    }
):
    if "__end__" not in s:
        print(s)
        print("----")
        
for s in graph.stream(
    {"messages": [HumanMessage(content="Write a brief research report on evolution of the Tetris game.")]},
    {"recursion_limit": 10},
):
    if "__end__" not in s:
        print(s)
        print("----")

     
for s in graph.stream(
    {"messages": [HumanMessage(content="Create the tetris game in python code, and return the code.")]},
    {"recursion_limit": 10},
):
    if "__end__" not in s:
        print(s)
        print("----")
        
result = graph.invoke(
    input = {"messages": [HumanMessage(content="Create the tetris game in python code, and return the code.")]},
    config = {"recursion_limit": 10},
)

pprint(result)

result['messages']

last_message = result['messages'][-1]

dict(last_message)

pprint(last_message.content)

       
result_2 = graph.invoke(
    input = {"messages": [HumanMessage(content="What is the historical temperature in Pittsburgh over the last 5 years. Plot in a chart.")]},
    config = {"recursion_limit": 20},
)

result_2

      
result_3 = graph.invoke(
    input = {"messages": [HumanMessage(content="Find the historical prices of SPY for the last 5 years from Yahoo Finance. Plot a line chart of the SPY value over time from the historical prices using python.")]},
    config = {"recursion_limit": 40},
)

result_3["messages"][-1].name

pprint(dict(result_3), width=40, compact=True)

for message in result_3['messages']:
    if message.name:
        print(f"Name: {message.name}")
    print(f"Content: {message.content}")
    print("---")
    print()
