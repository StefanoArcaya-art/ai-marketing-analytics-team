# *** MARKETING COPY-WRITER / EMAIL EXPERT (BASED ON PROMPT ENGINEERING FROM AI FAST TRACK) ***

# Key Modifications:
# 1. Implements Prompt Template and prompt engineering to create a marketing email copywriter persona.
# 2. Integrates "chat_history" to provide context for the agent based on the previous sequence of messages.
# 3. Returns a Compiled LangGraph app

# LIBRARIES

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage, AIMessage

from langgraph.graph import StateGraph, START, END

from typing import Sequence, TypedDict

from customer_analytics_team.agents.utils import get_last_human_message

# AGENT

def make_marketing_agent(model, temperature=0):
    
    # Handle case when users want to make a different model than ChatOpenAI
    if isinstance(model, str):
        llm = ChatOpenAI(model = model)
    else:
        llm = model
        
    llm.temperature = temperature
    
    # * Marketing Agent

    marketing_agent_prompt = PromptTemplate(
        template="""
        You are an expert in writing marketing email copy for Business Science, a premium data science educational platform. Don't use markdown headers in your response.
        
        Your emails are designed to inform customers about products that they might be interested in, and to target customers by email address.
        
        Your email should specify:
        
        1. Which customer email addresses to target
        2. An attention grabbing email subject
        3. Include email content designed to educate and provide reasons that they should invest in our educational program
        
        Examples of good reasons to purchase include:
        1. Getting a career advancement or new job
        2. Developing a portfolio to attract recruiters and wow their bosses
        3. Taking action: Stop procrastinating and finally take action towards a fullfilling career
        4. Increased Salary: Many Data Scientists make over $100,000 per year. 
        
        IMPORTANT: Make sure to explain why the product you are recommending will help them address their goal.
        
        INITIAL_USER_QUESTION: {initial_question}
        CONTEXT: {chat_history}
        """,
        input_variables=["initial_question", "chat_history"]
    )

    marketing_agent = marketing_agent_prompt | llm | StrOutputParser()
    
    # * CREATE AGENT LANGGRAPH WORKFLOW
    
    class GraphState(TypedDict):
        messages: Sequence[BaseMessage]
        response: Sequence[BaseMessage]
        
    def marketing_email_writer_node(state):
    
        print("---MARKETING EMAIL WRITER---")
        
        messages = state.get("messages")
        
        last_question = get_last_human_message(messages)
        if last_question:
            last_question = last_question.content
            
        result = marketing_agent.invoke({'initial_question': last_question,'chat_history': messages})
        
        return {
            "response": [AIMessage(content=result, name='Marketing_Email_Writer')],
        }
    
    workflow = StateGraph(GraphState)

    workflow.add_node("marketing_expert", marketing_email_writer_node)
    
    workflow.add_edge(START, "marketing_expert")
    workflow.add_edge("marketing_expert", END)
    
    app = workflow.compile()
    
    return app



