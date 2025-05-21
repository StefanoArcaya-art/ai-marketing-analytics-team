# BUSINESS SCIENCE UNIVERSITY
# PYTHON FOR GENERATIVE AI COURSE
# MULTI-AGENTS (AGENTIAL SUPERVISION)

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated, Sequence, TypedDict
import operator
import pandas as pd
from pprint import pprint
from IPython.display import Markdown, Image
import os
import yaml

# Sub-Agents
from marketing_analytics_team.agents.marketing_email_writer_agent import make_marketing_email_writer_agent
from marketing_analytics_team.agents.product_expert import make_product_expert_agent
from marketing_analytics_team.agents.business_intelligence_agent import make_business_intelligence_agent
from marketing_analytics_team.agents.segment_analysis_agent import make_segment_analysis_agent

def make_marketing_analytics_team_2(model, model_embedding, path_products_vector_db, path_transactions_sql_db, checkpointer):
    # API Keys
    os.environ["OPENAI_API_KEY"] = yaml.safe_load(open('../credentials.yml'))['openai']

    # STEP 1: MAKE SUPERVISOR AGENT
    def make_marketing_analytics_supervisor_agent(model, temperature=0):
        subagent_names = [
            "Product_Expert",
            "Business_Intelligence_Expert",
            "Marketing_Email_Writer",
            "Segment_Analysis_Agent"
        ]

        if isinstance(model, str):
            llm = ChatOpenAI(model=model)
        else:
            llm = model

        system_prompt = (
            """
            You are a supervisor tasked with managing a conversation between the following workers: {subagent_names}.

            Each worker has the following knowledge and skills:

                1. Product_Expert: Knows course content and can explain details from course sales pages.
                (Do not have the Product_Expert write emails—that’s the Marketing_Email_Writer’s job.)

                2. Business_Intelligence_Expert: Knows our customer transactions database.
                Can write SQL, produce tables and charts based on leads, purchases, and transactions data.

                3. Marketing_Email_Writer: Drafts marketing emails using Product_Expert content and
                customer segments identified by the Business_Intelligence_Expert or Segment_Analysis_Agent.

                4. Segment_Analysis_Agent: Analyzes precomputed customer segments in the database,
                generating descriptive labels and insights based on lead scores, engagement ratings,
                and purchase frequency.

            Assignment Rules:
            • Track which worker acted last.
            • **Never** assign the same worker twice in a row unless they explicitly request to continue.
            • If the same expertise is needed twice, see if a different worker can handle the follow-up (e.g., Business_Intelligence_Expert hands off to Product_Expert for context).
            • When multiple workers can fulfill a request, rotate in round-robin order to balance workload.

            Workflow:
            1. Read the user’s request.
            2. Decide which worker is best suited **and** is not the same as the last one you chose.
            3. Respond with exactly the worker’s name (e.g. `Business_Intelligence_Expert`) to invoke them.
            4. That worker will perform their task and return results.
            5. Repeat until the task is complete, then respond with `FINISH`.
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

    # STEP 2: INITIALIZE SUB-AGENTS
    supervisor_agent = make_marketing_analytics_supervisor_agent(model=model, temperature=0.7)

    product_expert_agent = make_product_expert_agent(
        model=model,
        model_embedding=model_embedding,
        db_path=path_products_vector_db
    )

    business_intelligence_agent = make_business_intelligence_agent(
        model=model,
        db_path=path_transactions_sql_db
    )

    marketing_agent = make_marketing_email_writer_agent(model=model)

    segment_analysis_agent = make_segment_analysis_agent(
        model=model,
        db_path=path_transactions_sql_db.replace("leads_scored.db", "leads_scored_segmentation.db"),
        temperature=0.7
    )

    # STEP 3: SUPERVISOR-LED TEAM LANGGRAPH
    class GraphState(TypedDict):
        # Team Messages State Tracking
        messages: Annotated[Sequence[BaseMessage], operator.add]
        # Supervisor State Tracking
        next: str
        # Business Intelligence Expert State Tracking
        sql_query: str
        data: dict
        chart_plotly_code: str
        chart_plotly_json: dict
        # Marketing Email Writer State Tracking
        email_list: list
        email_subject: str
        email_body: str
        # Segment Analysis Agent State Tracking
        segment_labels: dict
        segmentation_data: dict
        chart_json: str

    def supervisor_node(state):
        result = supervisor_agent.invoke(state)
        print(result.get("next"))
        return {'next': result.get("next")}

    def product_expert_node(state):
        result = product_expert_agent.invoke(state)
        return {"messages": result.get("response")}

    def business_intelligence_expert_node(state):
        result = business_intelligence_agent.invoke(state)
        return {
            "messages": result.get("response"),
            "sql_query": result.get("sql_query"),
            "data": result.get("data"),
            "chart_plotly_code": result.get("chart_plotly_code"),
            "chart_plotly_json": result.get("chart_plotly_json"),
        }

    def email_writer_node(state):
        result = marketing_agent.invoke(state)
        return {
            "messages": result.get("response"),
            "email_list": result.get("email_list"),
            "email_subject": result.get("email_subject"),
            "email_body": result.get("email_body"),
        }

    def segment_analysis_node(state):
        result = segment_analysis_agent.invoke(state)
        return {
            "messages": result.get("response"),
            "segment_labels": result.get("segment_labels"),
            "segmentation_data": result.get("segmentation_data"),
            "chart_json": result.get("chart_json"),
        }

    # WORKFLOW DAG
    workflow = StateGraph(GraphState)

    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("Product_Expert", product_expert_node)
    workflow.add_node("Business_Intelligence_Expert", business_intelligence_expert_node)
    workflow.add_node("Marketing_Email_Writer", email_writer_node)
    workflow.add_node("Segment_Analysis_Agent", segment_analysis_node)

    workflow.set_entry_point("supervisor")

    workflow.add_edge('Product_Expert', "supervisor")
    workflow.add_edge('Business_Intelligence_Expert', "supervisor")
    workflow.add_edge('Marketing_Email_Writer', "supervisor")
    workflow.add_edge('Segment_Analysis_Agent', "supervisor")

    workflow.add_conditional_edges(
        "supervisor",
        lambda state: state.get("next"),
        {
            'Product_Expert': 'Product_Expert',
            'Business_Intelligence_Expert': 'Business_Intelligence_Expert',
            'Marketing_Email_Writer': 'Marketing_Email_Writer',
            'Segment_Analysis_Agent': 'Segment_Analysis_Agent',
            'FINISH': END
        }
    )

    # Compile with short-term memory
    app = workflow.compile(checkpointer=checkpointer)

    return app

# Example usage (for testing)
if __name__ == "__main__":
    from langchain_core.messages import HumanMessage
    from IPython.display import Markdown, Image
    import plotly.io as pio

    # Key Inputs
    MODEL = 'gpt-4o-mini'
    EMBEDDINGS_MODEL = 'text-embedding-ada-002'
    PATH_PRODUCTS_VECTORDB = "data/data-rag-product-information/products_clean.db"
    PATH_TRANSACTIONS_DATABASE = "sqlite:///challenges/data/database-sql-transactions/leads_scored_segmentation.db"
    checkpointer = MemorySaver()

    # Initialize team
    marketing_analytics_team = make_marketing_analytics_team_2(
        model=MODEL,
        model_embedding=EMBEDDINGS_MODEL,
        path_products_vector_db=PATH_PRODUCTS_VECTORDB,
        path_transactions_sql_db=PATH_TRANSACTIONS_DATABASE,
        checkpointer=checkpointer
    )

    # Test: Segment Analysis
    messages = [HumanMessage(content="Can you analyze the customer segments and provide insights from the Segmentation Agent?")]
    result = marketing_analytics_team.invoke(
        input={"messages": messages},
        config={
            "recursion_limit": 10,
            "configurable": {"thread_id": "123"}
        },
    )

    print("Messages:")
    for message in result['messages']:
        if message.name:
            pprint(message.name)
        pprint(message.content)

    print("\nSegment Labels:")
    pprint(result['segment_labels'])

    print("\nSegmentation Data:")
    pd.DataFrame(result['segmentation_data'])

    # Display chart
    fig = pio.from_json(result['chart_json'])
    fig.show()

