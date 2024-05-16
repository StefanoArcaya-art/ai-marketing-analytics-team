# https://langchain-ai.github.io/langgraph/tutorials/multi_agent/multi-agent-collaboration/


import os
import functools
from pprint import pprint

from langchain_core.messages import AIMessage

from langchain_core.messages import (
    BaseMessage,
    ToolMessage,
    HumanMessage,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph

from langchain_core.tools import tool

from langgraph.prebuilt import ToolNode

from langchain_experimental.utilities import PythonREPL
from langchain_community.tools.tavily_search import TavilySearchResults

import operator
from typing import Annotated, Sequence, TypedDict, Literal

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

import yaml

# APIS 

os.environ['GROQ_API_KEY'] =  yaml.safe_load(open('../credentials.yml'))['groq']

os.environ["OPENAI_API_KEY"] = yaml.safe_load(open('../credentials.yml'))['openai']

llm = ChatOpenAI(model="gpt-3.5-turbo")
# llm = ChatGroq(model="llama3-70b-8192")

os.environ["TAVILY_API_KEY"] = yaml.safe_load(open("../credentials.yml"))['tavily']

tavily_tool = TavilySearchResults(max_results=5)


# CREATE AGENT

def create_agent(llm, tools, system_message: str):
    """Create an agent."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful AI assistant, collaborating with other assistants."
                " Use the provided tools to progress towards answering the question."
                " If you are unable to fully answer, that's OK, another assistant with different tools "
                " will help where you left off. Execute what you can to make progress."
                " If you or any of the other assistants have the final answer or deliverable,"
                " prefix your response with FINAL ANSWER so the team knows to stop."
                " You have access to the following tools: {tool_names}.\n{system_message}",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    return prompt | llm.bind_tools(tools)

# DEFINE TOOLS

repl = PythonREPL()

@tool
def python_repl(
    code: Annotated[str, "The python code to execute to generate your chart."]
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    try:
        print(code)
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    return (
        result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."
    )
    
# CREATE GRAPH

# 1.0 STATE

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str
    
# 2.0 AGENT NODES 

# Helper function to create a node for a given agent
def agent_node(state, agent, name):
    if 'input' not in state:
        state['input'] = state.get('content', '')  # Assuming 'content' is the query or similar relevant input
    result = agent.invoke(state)

    # Debugging to see what type of result is returned
    print("Debugging - Result type:", type(result))
    print("Debugging - Result content:", result)

    # Check if the result is an instance of ToolMessage
    if isinstance(result, ToolMessage):
        pass  # ToolMessage handling logic goes here

    # Check if the result is an instance of AIMessage directly
    elif isinstance(result, AIMessage):
        # Directly use the AIMessage if it's already in the correct format
        pass

    # Check if the result is a dictionary and convert it to AIMessage if needed
    elif isinstance(result, dict):
        # Ensuring that 'content' key exists in result or providing a default value
        content = result.get('content', 'No content provided')
        result = AIMessage(content=content, name=name, **result)

    else:
        # If result is neither ToolMessage, AIMessage, nor dict, handle appropriately
        raise ValueError(f"Unexpected result type from agent.invoke(): {type(result)} with content: {result}")

    return {
        "messages": [result],
        "sender": name,
    }


# SQL Agent
from langchain_community.agent_toolkits import create_sql_agent

from langchain_community.utilities import SQLDatabase

PATH_DB = "sqlite:///database/leads_scored.db"

db = SQLDatabase.from_uri(PATH_DB)
db.get_usable_table_names()
db.run("SELECT * FROM leads_scored LIMIT 10;")

sql_agent = create_sql_agent(
    llm, 
    db=db,
    agent_type="openai-tools"
)
sql_node = functools.partial(agent_node, agent=sql_agent, name="sql_agent")



# Research agent and node
research_agent = create_agent(
    llm,
    [tavily_tool],
    system_message="You should provide accurate data for the chart_generator to use.",
)
research_node = functools.partial(agent_node, agent=research_agent, name="Researcher")

# chart_generator
chart_agent = create_agent(
    llm,
    [python_repl],
    system_message="Any charts you display will be visible by the user.",
)
chart_node = functools.partial(agent_node, agent=chart_agent, name="chart_generator")

# 3.0 TOOL NODES

tools = [tavily_tool, python_repl]

tool_node = ToolNode(tools)


# EDGE LOGIC

def router(state) -> Literal["call_tool", "__end__", "continue"]:
    # This is the router
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        # The previous agent is invoking a tool
        return "call_tool"
    if "FINAL ANSWER" in last_message.content:
        # Any agent decided the work is done
        return "__end__"
    return "continue"

# DEFINE GRAPH
workflow = StateGraph(AgentState)

workflow.add_node("sql_agent", sql_node)
workflow.add_node("chart_generator", chart_node)
workflow.add_node("call_tool", tool_node)

workflow.add_conditional_edges(
    "sql_agent",
    router,
    {"continue": "chart_generator", "call_tool": "call_tool", "__end__": END},
)
workflow.add_conditional_edges(
    "chart_generator",
    router,
    {"continue": "sql_agent", "call_tool": "call_tool", "__end__": END},
)

workflow.add_conditional_edges(
    "call_tool",
    lambda x: x["sender"],
    {
        "sql_agent": "sql_agent",
        "chart_generator": "chart_generator",
    },
)
workflow.set_entry_point("sql_agent")

graph = workflow.compile()

from IPython.display import Image, display

try:
    display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
except:
    # This requires some extra dependencies and is optional
    pass

# TESTS MESSAGE

events = graph.stream(
    {
        "messages": [
            HumanMessage(
                content="Get the top 10 customers in sales, then draw a bar chart of their email address vs sales. Then finish."
            )
        ],
    },
    # Maximum number of steps to take in the graph
    {"recursion_limit": 150},
)
for s in events:
    pprint(s)
    print("----")
    
