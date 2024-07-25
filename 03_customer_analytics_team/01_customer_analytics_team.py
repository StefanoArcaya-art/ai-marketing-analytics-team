

# * LIBRARIES

from langchain_community.vectorstores import Chroma

from langchain_core.output_parsers import StrOutputParser, JsonOutputParser, BaseOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from langchain.prompts import PromptTemplate

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever

from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain

from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL

from langgraph.graph import StateGraph, END

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from typing import Annotated, Sequence, TypedDict

import operator

import pandas as pd
import sqlalchemy as sql

import plotly as pl
import plotly.express as px
import plotly.io as pio

import os
import yaml
import ast
import json
import re

from pprint import pprint
from IPython.display import Image

# * DATABASE SETUP

PATH_PRODUCTS_VECTORDB = "data/data-rag-product-information/products_clean.db"

PATH_TRANSACTIONS_DATABASE = "sqlite:///data/database-sql-transactions/leads_scored.db"


# * AI SETUP

os.environ["OPENAI_API_KEY"] = yaml.safe_load(open('../credentials.yml'))['openai']

OPENAI_LLM = ChatOpenAI(
    model = "gpt-4o-mini"
)

# *** SUPERVISOR AGENT ***

subagent_names = ["Product_Expert", "Customer_Transactions_Expert", "Marketing_Email_Writer"]

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
    
    vectorstore = Chroma(
        embedding_function=embedding_function, 
        persist_directory=db_path
    )

    retriever = vectorstore.as_retriever()
    
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


# *** CUSTOMER TRANSACTIONS EXPERT ***

def create_customer_transactions_agent(db_path, llm, temperature=0):    
    
    PATH_DB = db_path
    
    llm.temperature = temperature
    
    # * Handle Messages
    
    # * Routing Preprocessor Agent

    routing_preprocessor_prompt = PromptTemplate(
        template="""
        You are an expert in routing decisions for a SQL database agent, a Charting Visualization Agent, and a Pandas Table Agent. Your job is to:
        
        1. Determine what the correct format for a Users Question should be for use with a SQL translator agent 
        2. Determine whether or not a chart should be generated or a table should be returned based on the users question.
        
        Use the following criteria on how to route the the initial user question:
        
        From the incoming user question, remove any details about the format of the final response as either a Chart or Table and return only the important part of the incoming user question that is relevant for the SQL generator agent. This will be the 'formatted_user_question_sql_only'. If 'None' is found, return the original user question.
        
        Next, determine if the user would like a data visualization ('chart') or a 'table' returned with the results of the SQL query. If unknown, not specified or 'None' is found, then select 'table'.  
        
        Return JSON with 'formatted_user_question_sql_only' and 'routing_preprocessor_decision'.
        
        INITIAL_USER_QUESTION: {initial_question}
        """,
        input_variables=["initial_question"]
    )

    routing_preprocessor = routing_preprocessor_prompt | llm | JsonOutputParser()

    # * SQL Agent

    db = SQLDatabase.from_uri(PATH_DB)

    def extract_sql_code(text):
        sql_code_match = re.search(r'```sql(.*?)```', text, re.DOTALL)
        if sql_code_match:
            sql_code = sql_code_match.group(1).strip()
            return sql_code
        else:
            sql_code_match = re.search(r"sql(.*?)'", text, re.DOTALL)
            if sql_code_match:
                sql_code = sql_code_match.group(1).strip()
                return sql_code
            else:
                return None
        
    class SQLOutputParser(BaseOutputParser):
        def parse(self, text: str):
            sql_code = extract_sql_code(text)
            if sql_code is not None:
                return sql_code
            else:
                # Assume ```sql wasn't used
                return text

    prompt_sqlite = PromptTemplate(
        input_variables=['input', 'table_info', 'top_k'],
        template="""
        You are a SQLite expert. Given an input question, first create a syntactically correct SQLite query to run, then look at the results of the query and return the answer to the input question.
        
        Do not use a LIMIT clause with {top_k} unless a user specifies a limit to be returned.
        
        Return SQL in ```sql ``` format.
        
        Only return a single query if possible.
        
        Never query for all columns from a table unless the user specifies it. You must query only the columns that are needed to answer the question unless the user specifies it. Wrap each column name in double quotes (") to denote them as delimited identifiers.
        
        Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
        
        Pay attention to use date(\'now\') function to get the current date, if the question involves "today".
            
        Only use the following tables:
        {table_info}
        
        Question: {input}'
        """
    )

    sql_generator = (
        create_sql_query_chain(
            llm = llm,
            db = db,
            k = int(1e7),
            prompt = prompt_sqlite
        ) 
        | SQLOutputParser() # NEW SQLCodeExtactor
    )

    # * Dataframe Conversion
        
    sql_engine = sql.create_engine(PATH_DB)

    conn = sql_engine.connect()

    # * Chart Instructor

    prompt_chart_instructions = PromptTemplate(
        template="""
        You are a supervisor that is an expert in providing instructions to a chart generator agent for plotting. 
        
        You will take a question that a user has and the data that was generated to answer the question, and create instructions to create a chart from the data that will be passed to a chart generator agent.
        
        USER QUESTION: {question}
        
        DATA: {data}
        
        Formulate "chart generator instructions" by informing the chart generator of what type of plotly plot to use (e.g. bar, line, scatter, etc) to best represent the data. 
        
        Come up with an informative title from the user's question and data provided. Also provide X and Y axis titles.
        
        Return your instructions in the following format:
        CHART GENERATOR INSTRUCTIONS: FILL IN THE INSTRUCTIONS HERE
        
        """,
        input_variables=['question', 'data']
    )

    chart_instructor = prompt_chart_instructions | llm | StrOutputParser()


    # * Chart Generator

    repl = PythonREPL()

    @tool
    def python_repl(
        code: Annotated[str, "The python code to execute to generate your chart."]
    ):
        """Use this to execute python code. If you want to see the output of a value,
        you should print it out with `print(...)`. This is visible to the user."""
        try:
            result = repl.run(code)
        except BaseException as e:
            return f"Failed to execute. Error: {repr(e)}"
        return f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"


    prompt_chart_generator = PromptTemplate(
        template = """
        You are an expert in creating data visualizations and plots using the plotly python library. You must use plotly or plotly.express to produce plots.
        
        Your job is to produce python code to generate visualizations.
        
        Create the python code to produce the requested visualization given the plot requested from the original user question and the input data. 
        
        The input data will be provided as a dictionary and will need converted to a pandas data frame before creating the visualization. 
        
        The output of the plotly chart should be stored as a JSON object with pio.to_json() and then to a dictionary. 
        
        Make sure to add: import plotly.io as pio
        Make sure to print the fig_dict
        Make sure to import json
        
        Here's an example of converting a plotly object to JSON:
        
        import json
        import plotly.graph_objects as go
        import plotly.io as pio

        # Create a sample Plotly figure
        fig = go.Figure(data=go.Bar(y=[2, 3, 1]))

        # Convert the figure to JSON
        fig_json = pio.to_json(fig)
        fig_dict = json.loads(fig_json)
        
        print(fig_dict) # MAKE SURE TO DO THIS
        
        
        CHART INSTRUCTIONS: {chart_instructions}
        INPUT DATA: {data}
        
        Important Notes on creating the chart code:
        - Do not use color_discrete_map. This is an invalid property.
        - If bar plot, do not add barnorm='percent' unless user asks for it
        - If bar plot, do not add a trendline. Plotly bar charts do not natively support the trendline.  
        - For line plots, the line width should be updated on traces (example: # Update traces
    fig.update_traces(line=dict(color='#3381ff', width=0.65)))
        - For Bar plots, the default line width is acceptable
        - Super important - Make sure to print(fig_dict)
        """,
        input_variables=["chart_instructions", "data"]
    )

    tools = [python_repl]

    chart_generator = prompt_chart_generator.partial(tool_names=", ".join([tool.name for tool in tools])) | llm.bind_tools(tools)


    # * LANGGRAPH
    class GraphState(TypedDict):
        """
        Represents the state of our graph.
        """
        user_question: str
        formatted_user_question_sql_only: str
        sql_query : str
        data: dict
        routing_preprocessor_decision: str
        chart_generator_instructions: str
        chart_plotly_code: str
        chart_plotly_json: dict
        chart_plotly_error: bool
        num_steps : int
        
    def preprocess_routing(state):
        print("---ROUTER---")
        question = state.get("user_question")
        
        num_steps = state.get("num_steps")
        
        num_steps += 1
        
        # Chart Routing and SQL Prep
        response = routing_preprocessor.invoke({"initial_question": question})
        
        formatted_user_question_sql_only = response['formatted_user_question_sql_only']
        
        routing_preprocessor_decision = response['routing_preprocessor_decision']
        
        return {
            "formatted_user_question_sql_only": formatted_user_question_sql_only,
            "routing_preprocessor_decision": routing_preprocessor_decision,
            "num_steps": num_steps
        }
        


    def generate_sql(state):
        print("---GENERATE SQL---")
        question = state.get("formatted_user_question_sql_only")
        
        # Handle case when formatted_user_question_sql_only is None:
        if question is None:
            question = state.get("user_question")
        
        num_steps = state.get("num_steps")
        
        num_steps += 1
        
        # Generate SQL
        sql_query = sql_generator.invoke({"question": question})
        
        return {"sql_query": sql_query, "num_steps": num_steps}


    def convert_dataframe(state):
        print("---CONVERT DATA FRAME---")

        sql_query = state.get("sql_query")
        
        num_steps = state.get("num_steps")
        
        num_steps += 1
        
        # Remove trailing ' that gpt-3.5-turbo sometimes leaves
        sql_query = sql_query.rstrip("'")
        
        df = pd.read_sql(sql_query, conn)
        
        return {"data": dict(df), "num_steps": num_steps}


    def decide_chart_or_table(state):
        print("---DECIDE CHART OR TABLE---")
        return "chart" if state.get('routing_preprocessor_decision') == "chart" else "table"

    def instruct_chart_generator(state):
        print("---INSTRUCT CHART GENERATOR---")
        
        question = state.get("user_question")
        
        data = state.get("data")
        
        num_steps = state.get("num_steps")
        
        num_steps += 1
        
        chart_generator_instructions = chart_instructor.invoke({"question": question, "data": data})
        
        return {"chart_generator_instructions": chart_generator_instructions, "num_steps": num_steps}


    def generate_chart(state):
        print("---GENERATE CHART---")
        
        chart_instructions = state.get("chart_generator_instructions")
        
        data = state.get("data")
        
        num_steps = state.get("num_steps")
        
        num_steps += 1
        
        response = chart_generator.invoke({"chart_instructions": chart_instructions, "data": data})
        
        # Fix - if invalid tool calls
        try:
            code = dict(response)['tool_calls'][0]['args']['code']
        except: 
            code = dict(response)['invalid_tool_calls'][0]['args']
        
        result = repl.run(code)
        
        # Add plot to global environment
        # globals()["chart_plotly_json"] = result
        
        chart_plotly_error = False
        if "error" in result[:40].lower():
            chart_plotly_error = True
        else:
            try:
                result_dict = ast.literal_eval(result)
            
                fig = pio.from_json(json.dumps(result_dict))

                # fig.show()
            except:
                chart_plotly_error = True
            
        return {
            "chart_plotly_code": code, 
            "chart_plotly_json": result, 
            "chart_plotly_error": chart_plotly_error,
            "num_steps": num_steps,
        }
        
        
    def state_printer(state):
        """print the state"""
        print("---STATE PRINTER---")
        print(f"User Question: {state['user_question']}")
        print(f"Formatted Question (SQL): {state['formatted_user_question_sql_only']}")
        print(f"SQL Query: \n{state['sql_query']}\n")
        print(f"Data: \n{pd.DataFrame(state['data'])}\n")
        print(f"Chart or Table: {state['routing_preprocessor_decision']}")
        
        if state['routing_preprocessor_decision'] == "chart":
            print(f"Chart Code: \n{pprint(state['chart_plotly_code'])}")
            print(f"Chart Error: {state['chart_plotly_error']}")
        
        print(f"Num Steps: {state['num_steps']}")

    # * WORKFLOW DAG

    workflow = StateGraph(GraphState)

    workflow.add_node("preprocess_routing", preprocess_routing)
    workflow.add_node("generate_sql", generate_sql)
    workflow.add_node("convert_dataframe", convert_dataframe)
    workflow.add_node("instruct_chart_generator", instruct_chart_generator)
    workflow.add_node("generate_chart", generate_chart)
    workflow.add_node("state_printer", state_printer)

    workflow.set_entry_point("preprocess_routing")
    workflow.add_edge("preprocess_routing", "generate_sql")
    workflow.add_edge("generate_sql", "convert_dataframe")

    workflow.add_conditional_edges(
        "convert_dataframe", 
        decide_chart_or_table,
        {
            # Result : Step Name To Go To
            "chart":"instruct_chart_generator", # Path Chart
            "table":"state_printer" # Path State Printer
        }
    )

    workflow.add_edge("instruct_chart_generator", "generate_chart")
    workflow.add_edge("generate_chart", "state_printer")
    workflow.add_edge("state_printer", END)

    app = workflow.compile()
    
    return app

customer_transactions_agent = create_customer_transactions_agent(db_path=PATH_TRANSACTIONS_DATABASE, llm = OPENAI_LLM, temperature=0)

Image(customer_transactions_agent.get_graph().draw_mermaid_png())


QUESTION = """
What are the top 5 product sales revenue by product name? Make a donut chart. Use suggested price for the sales revenue and a unit quantity of 1 for all transactions.
"""
result = customer_transactions_agent.invoke({"user_question": QUESTION, "num_steps": 0})

result

result_dict = ast.literal_eval(result["chart_plotly_json"])
        
fig = pio.from_json(json.dumps(result_dict))

fig


# *** MARKETING EMAIL WRITER ***


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


def email_writer_node(state):
    
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
workflow.add_node("Marketing_Email_Writer", email_writer_node)

for member in subagent_names:
    workflow.add_edge(member, "supervisor")

conditional_map = {
    'Product_Expert': 'Product_Expert', 
    'Customer_Transactions_Expert': 'Customer_Transactions_Expert', 
    'Marketing_Email_Writer':'Marketing_Email_Writer',
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
