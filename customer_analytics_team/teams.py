


# LIBRARIES

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.messages import BaseMessage, AIMessage

from langgraph.graph import StateGraph, END

from langchain_openai import ChatOpenAI

from typing import Annotated, Sequence, TypedDict
import operator


from customer_analytics_team.agents.business_intelligence_agent import make_business_intelligence_agent
from customer_analytics_team.agents.marketing_email_writer_agent import make_marketing_email_writer_agent
from customer_analytics_team.agents.product_expert import make_product_expert_agent

from customer_analytics_team.agents.utils import get_last_human_message


# * TEAM CREATION

def make_customer_analytics_team(model, path_products_vector_db, path_transactions_sql_db):
    
    # Handle case when users want to make a different model than ChatOpenAI
    if isinstance(model, str):
        llm = ChatOpenAI(model = model)
    else:
        llm = model 
        
    # * CREATE SUB AGENTS
    
    product_expert_agent = make_product_expert_agent(model=llm, db_path=path_products_vector_db)
    business_intelligence_agent = make_business_intelligence_agent(model=llm, db_path=path_transactions_sql_db)
    marketing_agent = make_marketing_email_writer_agent(model=llm)
    
    # * CREATE SUPERVISOR
        
    subagent_names = ["Product_Expert", "Business_Intelligence_Expert", "Marketing_Email_Writer"]

    def make_supervisor_chain(subagent_names: list, llm, temperature=0):

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
        
        llm.temperature = temperature
        
        supervisor_chain = (
            prompt
            | llm.bind(functions=[function_def], function_call={"name": "route"})
            | JsonOutputFunctionsParser()
        )
        
        return supervisor_chain
    
    supervisor_agent = make_supervisor_chain(subagent_names=subagent_names, llm=llm, temperature=0.7)

    # * LANGGRAPH

    class GraphState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]
        final_email: str
        final_email_title: str
        email_list: list
        next: str
        
    def supervisor_node(state):
        
        print("---SUPERVISOR---")
        
        result = supervisor_agent.invoke(state)
        
        print(result)
        
        return {'next': result['next']}


    def product_expert_node(state):
        
        print("---PRODUCT EXPERT---")
        
        messages = state.get("messages")
        
        result = product_expert_agent.invoke({"messages": messages})
        
        return {
            "messages": [AIMessage(content=result['answer'], name='Product_Expert')]
        }
        
    def business_intelligence_expert_node(state):
        
        print("---BUSINESS INTELLIGENCE EXPERT---")
        
        messages = state.get("messages")

        last_question = get_last_human_message(messages)
        if last_question:
            last_question = last_question.content
        
        result = business_intelligence_agent.invoke({
            "user_question": last_question, 
            "chat_history": messages
        })
        
        return {
            "messages": [AIMessage(content=result['summary'], additional_kwargs=result, name='Business_Intelligence_Expert')],
        }


    def email_writer_node(state):
        
        print("---MARKETING EMAIL WRITER---")
        
        messages = state.get("messages")
        
        last_question = get_last_human_message(messages)
        if last_question:
            last_question = last_question.content
            
        result = marketing_agent.invoke({'initial_question': last_question,'chat_history': messages})
        
        # Final email
        
        return {
            "messages": [AIMessage(content=result, name='Marketing_Email_Writer')],
            "final_email": result,
        }

    # * WORKFLOW DAG

    workflow = StateGraph(GraphState)

    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("Product_Expert", product_expert_node)
    workflow.add_node("Business_Intelligence_Expert", business_intelligence_expert_node)
    workflow.add_node("Marketing_Email_Writer", email_writer_node)

    for member in subagent_names:
        workflow.add_edge(member, "supervisor")

    workflow.add_conditional_edges(
        "supervisor", 
        lambda state: state["next"], 
        {
            'Product_Expert': 'Product_Expert', 
            'Business_Intelligence_Expert': 'Business_Intelligence_Expert', 
            'Marketing_Email_Writer':'Marketing_Email_Writer',
            'FINISH': END
        }
    )

    workflow.set_entry_point("supervisor")

    app = workflow.compile()

    return app
    
    

