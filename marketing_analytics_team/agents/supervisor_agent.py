# *** MARKETING ANALYTICS SUPERVISOR AGENT

# GOAL: This supervisor agent manages the conversation between the Product Expert, Business Intelligence Expert, and Marketing Email Writer.

# LangChain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.messages import BaseMessage

from langchain_openai import ChatOpenAI

# LangGraph
from langgraph.graph import StateGraph, START, END
from typing import Sequence, TypedDict


# * MARKETING ANALYTICS SUPERVISOR

def make_marketing_analytics_supervisor_agent(model, temperature=0):
    
    subagent_names = ["Product_Expert", "Business_Intelligence_Expert", "Marketing_Email_Writer"]
    
    # Handle case when users want to make a different model than ChatOpenAI
    if isinstance(model, str):
        llm = ChatOpenAI(model = model)
    else:
        llm = model 
        
    llm.temperature = temperature

    system_prompt = (
        """
        You are a supervisor tasked with managing a conversation between the following workers:  {subagent_names}. 
        
        Each worker has the following knowledge and skills:
        
            1. Product_Expert: Can explain details of contents inside the courses from the course sales pages. Do not have the Product Expert write emails (the Marketing Expert should do this). 
            
            2. Business_Intelligence_Expert: Has knowledge of the company's customer transactions database. Has access to the customer SQL database that includes SQL tables containing information on customers, lead scores (how likely they are to buy), transactions, courses purchased, and types of products. Can write SQL, produce data in table and charts. 
            
            3. Marketing_Email_Writer: Is skilled at drafting marketing emails using information from the Product_Expert to help explain what's inside various products that may be of benefit to the customer. Uses SQL queries and data from the Business_Intelligence_Expert to target customers by their email address and products that they have not currently purchased.
        
        Given the following user request, respond with the worker to act next. 
        
        Each worker will perform a task and respond with their results and status. When finished, respond with FINISH.
        """
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
    
    supervisor_chain = (
        prompt
        | llm.bind(functions=[function_def], function_call={"name": "route"})
        | JsonOutputFunctionsParser()
    )
    
    # LangGraph State Graph
    
    class GraphState(TypedDict):
        messages: Sequence[BaseMessage]
        next: str

    def supervisor_node(state):
        print("---SUPERVISOR---")
        
        messages = state.get("messages")
        
        result = supervisor_chain.invoke({"messages": messages})
        
        next_worker = result.get("next")

        return {"next": next_worker}

    workflow = StateGraph(GraphState)

    workflow.add_node("supervisor", supervisor_node)
    workflow.add_edge(START, "supervisor")
    workflow.add_edge("supervisor", END)

    app = workflow.compile()

    return app
