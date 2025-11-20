# PYTHON FOR GENERATIVE AI COURSE
# MULTI-AGENTS (AGENTIAL SUPERVISION)
# ***

# GOAL: Make a Marketing Email Copy Writer based on the AI Fast Track with Prompt Engineering

# LIBRARIES

# Modular Agent
from marketing_analytics_team.agents.marketing_email_writer_agent import make_marketing_email_writer_agent

# Agent Messages
from langchain_core.messages import HumanMessage, AIMessage

# Other Libraries
import os
import yaml
from pprint import pprint
from IPython.display import Markdown

# Backup to display mermaid graphs
from IPython.display import display, Image

# Key Inputs
MODEL = 'gpt-4.1-mini'

os.environ["OPENAI_API_KEY"] = yaml.safe_load(open('../credentials.yml'))['openai']

# * STEP 1: MAKE THE MARKETING COPY WRITER AGENT
#   * NEW: Refer to marketing_analytics_team.agents.marketing_email_writer_agent

marketing_agent = make_marketing_email_writer_agent(model=MODEL)

marketing_agent

# display(Image(marketing_agent.get_graph().draw_png()))

marketing_agent.get_input_jsonschema()['properties']


# * TESTS

# * TEST: Email with structured output

messages = [
    HumanMessage("Write a marketing email to customers who have not purchased 'Learning Labs Pro - Paid Course'")
]
result = marketing_agent.invoke({"messages": messages})

list(result.keys())

Markdown(result['response'][0].content)

pprint(result['response'][0].content)

result['email_subject']

Markdown(result['email_body'])

result['email_list']

# * TEST: Business Intelligence Agent includes email list

messages = [
    HumanMessage("Write a marketing email to customers who have not purchased 'Learning Labs Pro - Paid Course'"),
    AIMessage("The email list is: ['james_simons@rentech.com', 'bill_nye@scienceguy.com']", role="business_intelligence_agent")
]
result = marketing_agent.invoke({"messages": messages})

list(result.keys())

Markdown(result['response'][0].content)

pprint(result['response'][0].content)

result['email_subject']

Markdown(result['email_body'])

result['email_list']

# * TEST: No email requested

messages = [
    HumanMessage("What is the best way to promote courses in copy? No email is required.")
]
result = marketing_agent.invoke({"messages": messages})

Markdown(result['response'][0].content)