# *** BUSINESS INTELLIGENCE EXPERT (CLINIC #2) ***


from langchain_core.output_parsers import StrOutputParser, JsonOutputParser, BaseOutputParser
from langchain.prompts import PromptTemplate

from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain

from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL

from langgraph.graph import StateGraph, END

from langchain_openai import ChatOpenAI

from typing import Annotated, TypedDict


import pandas as pd
import sqlalchemy as sql
import plotly.io as pio

import ast
import json
import re

from pprint import pprint


def make_business_intelligence_agent(model, db_path):    
    
    # Handle case when users want to make a different model than ChatOpenAI
    if isinstance(model, str):
        llm = ChatOpenAI(model = model)
    else:
        llm = model 
    
    PATH_DB = db_path
    
    
    llm.temperature = 0
    
    # * Routing Preprocessor Agent

    routing_preprocessor_prompt = PromptTemplate(
        template="""
        You are an expert in routing decisions for a SQL database agent, a Charting Visualization Agent, and a Pandas Table Agent. Your job is to:
        
        1. Determine what the correct format for a Users Question should be for use with a SQL translator agent 
        2. Determine whether or not a chart should be generated or a table should be returned based on the users question.
        
        Use the following criteria on how to route the the initial user question:
        
        From the incoming user question and the chat_history, remove any details about the format of the final response as either a Chart or Table and return only the important part of the incoming user question that is relevant for the SQL generator agent. This will be the 'formatted_user_question_sql_only'. If 'None' is found, return the original user question.
        
        Next, determine if the user would like a data visualization ('chart') or a 'table' returned with the results of the SQL query. If unknown, not specified or 'None' is found, then select 'table'.  
        
        Return JSON with 'formatted_user_question_sql_only' and 'routing_preprocessor_decision'.
        
        INITIAL_USER_QUESTION: {initial_question}
        CONTEXT: {chat_history}
        """,
        input_variables=["initial_question", "chat_history"]
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
    
    # * Summarizer
    
    summarizer_prompt = PromptTemplate(
        template="""
        You are an expert in summarizing the analysis results of a Customer Transactions Expert. Your goal is to help the business understand the analysis in basic terms that business people can easily understand. Be consice in your explanation of the results. When possible please include summary tables to convey information instead of bullet points. Do not use markdown headers in your response.
        
        The Customer Transactions Expert as knowledge of the company's customer transactions database. Has analytics and business intelligence skills. Can write SQL, produce data in table and charts. Has access to the customer SQL database that includes SQL tables containing information on customers, lead scores (how likely they are to buy), transactions, courses purchased, and types of products.
        
        You are given the results of a the Customer Transaction Expert's analysis that contain:
        
        - user_question: The initial user question that was asked to the Customer Transactions Expert
        - chat_history: The previous chat history provided for additional context on the user's question
        - formatted_user_question_sql_only: A processed version of the user question provided to the SQL expert
        - sql_query: The sql query that the SQL expert created by accessing the Customer Analytics Database
        - data: The results of the sql query when run on the database
        - routing_processor_decision: either 'table' or 'chart'. If table, a chart is not returned. If chart, plotly code is created. 
        
        If a 'chart' was determined, the application will attempt to produce a chart. Sometimes errors occur, which is denoted by: 'chart_plotly_error'
        
        If a chart was successful, Python code and JSON will be produced, which are contained in 'chart_plotly_code' and 'chart_plotly_json' respectively. 
         
        ANALYSIS RESULTS FOR SUMMARIZATION: {results}
        """,
        input_variables=["results"]
    )

    summarizer = summarizer_prompt | llm | StrOutputParser()


    # * LANGGRAPH
    class GraphState(TypedDict):
        """
        Represents the state of our graph.
        """
        user_question: str
        chat_history: list
        formatted_user_question_sql_only: str
        sql_query : str
        data: dict
        routing_preprocessor_decision: str
        chart_generator_instructions: str
        chart_plotly_code: str
        chart_plotly_json: dict
        chart_plotly_error: bool
        summary: str
        
    def preprocess_routing(state):
        print("---ROUTER---")
        question = state.get("user_question")
        
        chat_history = state.get("chat_history")
        
        # Chart Routing and SQL Prep
        response = routing_preprocessor.invoke({"initial_question": question, "chat_history": chat_history})
        
        formatted_user_question_sql_only = response['formatted_user_question_sql_only']
        
        routing_preprocessor_decision = response['routing_preprocessor_decision']
        
        return {
            "formatted_user_question_sql_only": formatted_user_question_sql_only,
            "routing_preprocessor_decision": routing_preprocessor_decision,
        }

    def generate_sql(state):
        print("---GENERATE SQL---")
        question = state.get("formatted_user_question_sql_only")
        
        # Handle case when formatted_user_question_sql_only is None:
        if question is None:
            question = state.get("user_question")
        
        # Generate SQL
        sql_query = sql_generator.invoke({"question": question})
        
        return {"sql_query": sql_query}


    def convert_dataframe(state):
        print("---CONVERT DATA FRAME---")

        sql_query = state.get("sql_query")
        
        # Remove trailing ' that gpt-3.5-turbo sometimes leaves
        sql_query = sql_query.rstrip("'")
        
        df = pd.read_sql(sql_query, conn)
        
        return {"data": dict(df)}


    def decide_chart_or_table(state):
        print("---DECIDE CHART OR TABLE---")
        return "chart" if state.get('routing_preprocessor_decision') == "chart" else "table"

    def instruct_chart_generator(state):
        print("---INSTRUCT CHART GENERATOR---")
        
        question = state.get("user_question")
        
        data = state.get("data")
        
        chart_generator_instructions = chart_instructor.invoke({"question": question, "data": data})
        
        return {"chart_generator_instructions": chart_generator_instructions}


    def generate_chart(state):
        print("---GENERATE CHART---")
        
        chart_instructions = state.get("chart_generator_instructions")
        
        data = state.get("data")
        
        response = chart_generator.invoke({"chart_instructions": chart_instructions, "data": data})
        
        # Fix - if invalid tool calls
        try:
            code = dict(response)['tool_calls'][0]['args']['code']
        except: 
            code = dict(response)['invalid_tool_calls'][0]['args']
        
        result = repl.run(code)
        
        chart_plotly_error = False
        if "error" in result[:40].lower():
            chart_plotly_error = True
        else:
            try:
                result_dict = ast.literal_eval(result)
            
                fig = pio.from_json(json.dumps(result_dict))
            except:
                chart_plotly_error = True
            
        return {
            "chart_plotly_code": code, 
            "chart_plotly_json": result, 
            "chart_plotly_error": chart_plotly_error,
        }
        
    def summarize_results(state):
        print("---SUMMARIZE RESULTS----")
        
        result = summarizer.invoke({"results": dict(state)})
        
        return {"summary": result}
        
        
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
        
        

    # * WORKFLOW DAG

    workflow = StateGraph(GraphState)

    workflow.add_node("preprocess_routing", preprocess_routing)
    workflow.add_node("generate_sql", generate_sql)
    workflow.add_node("convert_dataframe", convert_dataframe)
    workflow.add_node("instruct_chart_generator", instruct_chart_generator)
    workflow.add_node("generate_chart", generate_chart)
    workflow.add_node("summarizer", summarize_results)
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
            "table":"summarizer" # Summarizer
        }
    )

    workflow.add_edge("instruct_chart_generator", "generate_chart")
    workflow.add_edge("generate_chart", "summarizer")
    workflow.add_edge("summarizer", "state_printer")
    workflow.add_edge("state_printer", END)

    app = workflow.compile()
    
    return app