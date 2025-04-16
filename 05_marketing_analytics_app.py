# BUSINESS SCIENCE UNIVERSITY
# PYTHON FOR GENERATIVE AI COURSE
# AI MARKETING ANALYTICS AGENT
# ***

# GOAL: Streamlit chat interface for Marketing Analytics Team
# Tracks: email_subject, email_body, email_list, plots/data, intermediate responses

# Command Line:
#   streamlit run app.py

# LIBRARIES ----
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.io as pio
import warnings

from langchain_core.messages import HumanMessage

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

from marketing_analytics_team.teams import make_marketing_analytics_team


warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

# Key Inputs:
CHAT_LLM_OPTIONS = ["gpt-4.1-nano", "gpt-4.1-mini", "gpt-4.1", "gpt-4o-mini", "gpt-4o"]
EMBEDDING_OPTIONS = ["text-embedding-ada-002"]
PATH_TRANSACTIONS_DB = "sqlite:///data/database-sql-transactions/leads_scored.db"
PATH_PRODUCTS_VDB   = "data/data-rag-product-information/products_clean.db"

# * STREAMLIT APP SETUP ----
st.set_page_config(page_title="Your AI Marketing Analytics Copilot", page_icon=":bar_chart:", layout="wide")
st.title("Your AI Marketing Analytics Agent")

with st.expander("I'm a complete marketing analytics copilot that contains a team of experts: Business intelligence SQL Expert, Product Expert, Marketing email writer. (see example questions)"):
    st.markdown(
        """
        - **Product Expert:** What are the key features of the 5-Course R-Track? 
        - **Business Intelligence Expert:** What are the top 10 customers by revenue? Draw a bar chart.
        - **Full Marketing Team with Email Writer**: Find the top 20 email subscribers ranked by probability of purchase (p1 lead score in the leads_scored table) who have have not purchased any courses yet? Have the Product Expert collect information on the 5-Course R-Track for use with the Marketing Expert. Have the Marketing Expert write a compelling marketing email.
        """
    )

# * Sidebar: Model & Database Selection ----

model_option = st.sidebar.selectbox("Choose OpenAI model", CHAT_LLM_OPTIONS)
embed_option = st.sidebar.selectbox("Choose embedding model", EMBEDDING_OPTIONS)

# * Initialize LLM & Team ----
llm = ChatOpenAI(model=model_option)
embeddings = OpenAIEmbeddings(model=embed_option)

marketing_team = make_marketing_analytics_team(
    model=llm,
    model_embedding=embeddings,
    path_products_vector_db=PATH_PRODUCTS_VDB,
    path_transactions_sql_db=PATH_TRANSACTIONS_DB,
    add_short_term_memory=True,
)

# * Chat Memory & Session State ----
msgs = StreamlitChatMessageHistory(key="marketing_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help with your marketing analytics today?")

for key in ["plots", "dataframes", "email_subjects", "email_bodies", "email_lists"]:
    if key not in st.session_state:
        st.session_state[key] = []

# * Display Function ----
def display_chat_history():
    for msg in msgs.messages:
        with st.chat_message(msg.type):
            c = msg.content
            if c.startswith("PLOT_INDEX:"):
                idx = int(c.split(":")[1])
                st.plotly_chart(st.session_state.plots[idx])
            elif c.startswith("DATAFRAME_INDEX:"):
                idx = int(c.split(":")[1])
                st.dataframe(st.session_state.dataframes[idx])
            elif c.startswith("EMAIL_SUBJECT_INDEX:"):
                idx = int(c.split(":")[1])
                st.markdown("**Email Subject:**")
                st.write(st.session_state.email_subjects[idx])
            elif c.startswith("EMAIL_BODY_INDEX:"):
                idx = int(c.split(":")[1])
                st.markdown("**Email Body:**")
                st.write(st.session_state.email_bodies[idx])
            elif c.startswith("EMAIL_LIST_INDEX:"):
                idx = int(c.split(":")[1])
                st.markdown("**Email List:**")
                st.write(st.session_state.email_lists[idx])
            else:
                st.write(c)

display_chat_history()

# * User Input & Invocation ----
if prompt := st.chat_input("Enter your marketing analytics request here…"):
    # Show the user question in chat
    st.chat_message("human").write(prompt)
    msgs.add_user_message(prompt)

    with st.spinner("Thinking…"):
        try:
            result = marketing_team.invoke(
                input={"messages": msgs.messages},
                config={"recursion_limit": 10, "configurable": {"thread_id": "marketing_thread"}}
            )
            error = False
        except Exception as e:
            error = True
            st.error("Oops—something went wrong with the team invocation.")
            print(e)

    if not error:
        # 1. Show all intermediate AI/Human messages
        #    (skip the very first one, since we already added the prompt)
        for m in result["messages"][1:]:
            if isinstance(m, HumanMessage):
                msgs.add_user_message(m.content)
            else:
                msgs.add_ai_message(m.content)

        # 2. Capture & display any plotly chart
        if result.get("chart_plotly_json"):
            fig = pio.from_json(result.get("chart_plotly_json"))
            i = len(st.session_state.plots)
            st.session_state.plots.append(fig)
            msgs.add_ai_message(f"PLOT_INDEX:{i}")

        # 3. Capture & display any tabular data
        elif result.get("data"):
            df = pd.DataFrame(result.get("data"))
            i = len(st.session_state.dataframes)
            st.session_state.dataframes.append(df)
            msgs.add_ai_message(f"DATAFRAME_INDEX:{i}")

        # 4. Email outputs
        #    Subject
        subj_idx = len(st.session_state.email_subjects)
        st.session_state.email_subjects.append(result.get("email_subject"))
        msgs.add_ai_message(f"EMAIL_SUBJECT_INDEX:{subj_idx}")
        #    Body
        body_idx = len(st.session_state.email_bodies)
        st.session_state.email_bodies.append(result.get("email_body"))
        msgs.add_ai_message(f"EMAIL_BODY_INDEX:{body_idx}")
        #    List
        list_idx = len(st.session_state.email_lists)
        st.session_state.email_lists.append(result.get("email_list"))
        msgs.add_ai_message(f"EMAIL_LIST_INDEX:{list_idx}")

        # 5. Rerender everything
        display_chat_history()
