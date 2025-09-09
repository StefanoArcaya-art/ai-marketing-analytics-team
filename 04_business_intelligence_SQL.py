# BUSINESS SCIENCE UNIVERSITY
# PYTHON FOR GENERATIVE AI COURSE
# MULTI-AGENTS (AGENTIAL SUPERVISION)
# ***

# GOAL: Make a business intelligence AI agent based on the SQL + Charting Agent from Clinic #2

# LIBRARIES

# Modular Agent
from marketing_analytics_team.agents.business_intelligence_agent import make_business_intelligence_agent

# Agent Messages
from langchain_core.messages import HumanMessage

# Other Libraries
import pandas as pd
import plotly.io as pio
import os
import yaml
from pprint import pprint
from IPython.display import Markdown

# Backup to display mermaid graphs
from IPython.display import display, Image

# Key Inputs
MODEL = 'gpt-4.1-mini'
PATH_CRM_DATABASE = "sqlite:///data/database-sql-transactions/leads_scored_segmentation.db"

os.environ["OPENAI_API_KEY"] = yaml.safe_load(open('../credentials.yml'))['openai']

# * STEP 1: MAKE THE BUSINESS INTELLIGENCE AGENT
# Key Modifications:
# 1. Routing Preprocessor Agent: Now gets chat_history
# 2. Summarizer Node: Added to summarize the analysis results, summary gets returned to the Supervisor Agent
# * Refer to marketing_analytics_team.agents.business_intelligence_agent for details

business_intelligence_agent = make_business_intelligence_agent(
    model=MODEL,
    db_path=PATH_CRM_DATABASE,
)

business_intelligence_agent

# display(Image(business_intelligence_agent.get_graph().draw_png()))

state_schema = business_intelligence_agent.get_input_jsonschema()['properties']

list(state_schema.keys())

# * TEST: Which 10 customers have the highest p1 probability of purchase who have NOT purchased 'Learning Labs Pro - Paid Course'? 

messages = [
    HumanMessage("Which 10 customers have the highest p1 probability of purchase who have NOT purchased 'Learning Labs Pro - Paid Course'?")
]

result = business_intelligence_agent.invoke({"messages": messages})

list(result.keys())

# Summary
Markdown(result['response'][0].content)

pprint(result['response'][0].content)

# SQL Query
pprint(result['sql_query'])

# Data
pd.DataFrame(result['data'])

dict(result['response'][0])['name']


# * TEST: What is the average P1 probability of purchase by member rating? Return a scatter chart with the results.
messages = [
    HumanMessage("What is the average P1 probability of purchase by member rating? Return a scatter chart with the results.")
]

result = business_intelligence_agent.invoke({"messages": messages})

list(result.keys())

# Summary
Markdown(result['response'][0].content)

pprint(result['response'][0].content)

# SQL Query
pprint(result['sql_query'])

# Plotly Chart
result['chart_plotly_json']

pio.from_json(result['chart_plotly_json'])
