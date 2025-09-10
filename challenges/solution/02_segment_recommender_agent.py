# BUSINESS SCIENCE UNIVERSITY
# PYTHON FOR GENERATIVE AI COURSE
# MULTI-AGENTS (AGENTIAL SUPERVISION)
# ***

# GOAL: Make a segment-aware recommender AI agent similar in style to the Customer Segmentation Agent (Part 2).
# The agent uses precomputed segments and past transactions to recommend a product per lead.

# * PART 2: SEGMENT RECOMMENDER AGENT

# AGENTS
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from typing import Sequence, TypedDict
import os
import yaml

# DATA SCIENCE
import pandas as pd
from sqlalchemy import create_engine
import plotly.express as px
import plotly.io as pio

# DEBUGGING
from pprint import pprint
from IPython.display import Markdown, Image, display

# CUSTOM
from marketing_analytics_team.agents.utils import get_last_human_message

# --------------------
# SETUP
# --------------------

db_path = "sqlite:///data/database-sql-transactions/leads_scored_segmentation.db"
model = "gpt-4.1-mini"

os.environ["OPENAI_API_KEY"] = yaml.safe_load(open('../credentials.yml'))['openai']
llm = ChatOpenAI(model=model, temperature=0)

# --------------------
# * 2.1 TESTING THE LLM
# --------------------

response = llm.invoke("What's the recipe for a margarita?")
pprint(response.content)
Markdown(response.content)  # Just like chatgpt!


# --------------------
# * 2.2 MAKING THE CUSTOM AGENT - SEGMENT RECOMMENDER
# --------------------

# Constraint Extraction Prompt (keeps the agent simple & controllable)
constraints_prompt = PromptTemplate(
    template="""
    Extract constraints for a segment-aware product recommender and return STRICT JSON:

    {{
      "top_n": integer (default 50),
      "exclude_corporate": boolean (default false),
      "exclude_domains": [list of domains like "acme.com"]
    }}

    INITIAL_USER_QUESTION: {initial_question}
    CONTEXT: {chat_history}
    """,
    input_variables=["initial_question", "chat_history"]
)

constraints_parser = constraints_prompt | llm | JsonOutputParser()

FREE_MAIL_PROVIDERS = {
    "gmail.com","yahoo.com","outlook.com","hotmail.com","icloud.com","aol.com","live.com","proton.me"
}
LOOKBACK_DAYS = 365
TOPK_PER_SEGMENT = 3

# Graph State
class GraphState(TypedDict):
    messages: Sequence[BaseMessage]
    response: Sequence[BaseMessage]
    reco_list: list
    chart_json: str
    constraints: dict
    sql_query: str


def segment_recommender_node(state: GraphState) -> GraphState:
    print("---SEGMENT RECOMMENDER AGENT---")

    # --------------------
    # 1) Connect to database & load tables
    # --------------------
    engine = create_engine(db_path)
    conn = engine.connect()

    leads_query = """
    SELECT user_email, p1, segment, made_purchase
    FROM leads_scored
    """

    transactions_query = """
    SELECT transaction_id, purchased_at, user_email, product_id
    FROM transactions
    """

    products_query = """
    SELECT product_id, description, suggested_price
    FROM products
    """

    df_leads = pd.read_sql(leads_query, conn)
    df_txn   = pd.read_sql(transactions_query, conn)
    df_prod  = pd.read_sql(products_query, conn)
    conn.close()

    # --------------------
    # 2) Timestamp standardization & lookback filter (robust to tz)
    # --------------------
    if "purchased_at" in df_txn.columns:
        s = pd.to_datetime(df_txn["purchased_at"], errors="coerce", utc=True)
        df_txn["purchased_at"] = s.dt.tz_convert(None)  # tz-naive
        cutoff = (pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=LOOKBACK_DAYS))
        df_txn = df_txn[df_txn["purchased_at"] >= cutoff]

    # --------------------
    # 3) Build segment → top-K product options (Part 1 logic, inline)
    # --------------------
    # Join transactions → segment
    df_txn_seg = df_txn.merge(df_leads[["user_email", "segment"]], on="user_email", how="left")
    df_txn_seg["segment"] = df_txn_seg["segment"].fillna(0).astype(int)

    # Join products for names/prices
    df_txn_seg = df_txn_seg.merge(
        df_prod[["product_id", "description", "suggested_price"]],
        on="product_id",
        how="left"
    )
    df_txn_seg["description"] = df_txn_seg["description"].fillna(df_txn_seg["product_id"].astype(str))
    df_txn_seg["suggested_price"] = df_txn_seg["suggested_price"].fillna(0.0)

    # Aggregate per segment×product
    agg = (
        df_txn_seg
        .groupby(["segment", "product_id", "description"], as_index=False)
        .agg(
            purchase_count=("transaction_id","count"),
            revenue_proxy=("suggested_price","sum"),
            avg_price=("suggested_price","mean")
        )
    )
    agg = agg.sort_values(["segment","purchase_count","revenue_proxy"], ascending=[True,False,False])
    agg["rank_in_segment"] = agg.groupby("segment").cumcount() + 1
    topk = agg[agg["rank_in_segment"] <= TOPK_PER_SEGMENT].copy()

    # Build map: segment -> list of dicts
    segment_to_products = (
        topk.sort_values(["segment", "rank_in_segment"])
            .groupby("segment")
            .apply(lambda g: [
                {
                    "product_id": float(pid) if pd.notna(pid) else None,
                    "product_name": str(desc),
                    "suggested_price": float(avg) if pd.notna(avg) else 0.0
                }
                for pid, desc, avg in zip(g["product_id"], g["description"], g["avg_price"])
            ])
            .to_dict()
    )

    # --------------------
    # 4) Parse constraints from the last human message (with defaults)
    # --------------------
    messages = state.get("messages", [])
    last_question = get_last_human_message(messages)
    last_question = last_question.content if last_question else ""

    default_constraints = {"top_n": 50, "exclude_corporate": False, "exclude_domains": []}
    try:
        parsed = constraints_parser.invoke({
            "initial_question": last_question,
            "chat_history": messages
        })
        # Merge defaults with parsed
        constraints = {**default_constraints, **{k:v for k,v in parsed.items() if v is not None}}
    except Exception as e:
        print("Constraint parse warning:", e)
        constraints = default_constraints

    top_n = int(constraints.get("top_n", 50))
    exclude_corporate = bool(constraints.get("exclude_corporate", False))
    exclude_domains = set((constraints.get("exclude_domains") or []))

    # --------------------
    # 5) Select top-N non-buyers by p1 & apply exclusions
    # --------------------
    leads = df_leads.copy()
    if "made_purchase" in leads.columns:
        nonbuyers = leads[leads["made_purchase"].fillna(0) == 0].copy()
    else:
        any_buy = df_txn.groupby("user_email").size().reset_index(name="txn_count")
        nonbuyers = leads.merge(any_buy, on="user_email", how="left")
        nonbuyers = nonbuyers[nonbuyers["txn_count"].fillna(0) == 0].copy()

    nonbuyers["segment"] = nonbuyers["segment"].fillna(0).astype(int)

    # Domain filters
    def _domain(email: str) -> str:
        x = str(email).split("@")
        return x[1].lower() if len(x) == 2 else ""

    domains = nonbuyers["user_email"].map(_domain)
    if exclude_corporate:
        nonbuyers = nonbuyers[domains.isin(FREE_MAIL_PROVIDERS)]
    if exclude_domains:
        nonbuyers = nonbuyers[~domains.isin({d.lower() for d in exclude_domains})]

    # Rank by p1
    nonbuyers = nonbuyers.sort_values("p1", ascending=False).head(top_n)

    # --------------------
    # 6) Recommend one product per lead (round-robin within the segment’s top-K)
    # --------------------
    # Global fallback options if a segment has no items
    overall_options = [p for arr in segment_to_products.values() for p in arr]
    if not overall_options and len(df_prod) > 0:
        overall_options = [{
            "product_id": float(df_prod.iloc[0]["product_id"]),
            "product_name": str(df_prod.iloc[0]["description"]),
            "suggested_price": float(df_prod.iloc[0]["suggested_price"] or 0.0)
        }]

    recos = []
    rr_ptr = {}  # segment -> pointer
    for _, row in nonbuyers.iterrows():
        seg = int(row["segment"])
        options = segment_to_products.get(seg, overall_options) or overall_options
        i = rr_ptr.get(seg, 0) % len(options)
        choice = options[i]
        rr_ptr[seg] = i + 1

        recos.append({
            "user_email": row["user_email"],
            "segment": seg,
            "product_id": choice["product_id"],
            "product_name": choice["product_name"],
            "expected_revenue": round(float(choice.get("suggested_price") or 0.0), 2)
        })

    # --------------------
    # 7) Visualization: donut of expected revenue by product (Top 5)
    # --------------------
    if len(recos) == 0:
        fig = px.pie(values=[1], names=["No Recos"], hole=0.5, title="Expected Revenue by Product")
    else:
        df_reco = pd.DataFrame(recos)
        donut = (
            df_reco.groupby("product_name")["expected_revenue"]
                   .sum().reset_index()
                   .sort_values("expected_revenue", ascending=False)
                   .head(5)
        )
        fig = px.pie(donut, values="expected_revenue", names="product_name", hole=0.5,
                     title="Expected Revenue by Product (Top 5)")
    chart_json = fig.to_json()

    # --------------------
    # 8) SQL provenance (for audit/debug)
    # --------------------
    sql_used = (
        "SELECT user_email, p1, segment, made_purchase FROM leads_scored; "
        "SELECT transaction_id, purchased_at, user_email, product_id FROM transactions; "
        "SELECT product_id, description, suggested_price FROM products;"
    )

    # --------------------
    # 9) Human-readable response
    # --------------------
    formatted_response = (
        f"**Segment Recommender Result**\n\n"
        f"- Constraints: top_n={top_n}, exclude_corporate={exclude_corporate}, "
        f"exclude_domains={sorted(list(exclude_domains))}\n"
        f"- Leads recommended: {len(recos)}\n"
        f"- Segments covered: {sorted(set([r['segment'] for r in recos])) if recos else []}\n"
        f"- Note: Segment options derived from last {LOOKBACK_DAYS} days of transactions."
    )

    return {
        # Like your segmentation agent: return an AIMessage with reasoning/summary
        "response": [AIMessage(content=formatted_response, name="SegmentRecommenderAgent")],

        # Core artifacts
        "reco_list": recos,
        "chart_json": chart_json,
        "constraints": constraints,
        "sql_query": sql_used,
    }


# Create LangGraph workflow
workflow = StateGraph(GraphState)
workflow.add_node("segment_recommender", segment_recommender_node)
workflow.add_edge(START, "segment_recommender")
workflow.add_edge("segment_recommender", END)

app = workflow.compile()
app

# --------------------
# * 2.3 RUNNING THE RECOMMENDER AGENT
# --------------------

messages = [HumanMessage(content="Recommend products for the top 50 non-buyers by p1. Exclude corporate domains.")]

results = app.invoke({"messages": messages})

list(results.keys())

# Messages and response
results['messages']
results['response']

# Display the AI message response
Markdown(results['response'][0].content)

# Display the chart
fig = pio.from_json(results['chart_json'])
fig

# Display first few recommendations
pd.DataFrame(results.get("reco_list")).head(10)


# --------------------
# * 2.4 MODULARIZE THE AGENT FOR THE MARKETING ANALYTICS TEAM
# --------------------
# Move the node logic above into:
#    marketing_analytics_team/agents/segment_recommender_agent.py
# and expose a factory: make_segment_recommender_agent(model, db_path)
#
# Example usage after creating that module:

# from marketing_analytics_team.agents.segment_recommender_agent import make_segment_recommender_agent
# segment_recommender_agent = make_segment_recommender_agent(model=model, db_path=db_path)
# segment_recommender_agent
# segment_recommender_agent.get_input_jsonschema()['properties']
# results = segment_recommender_agent.invoke({"messages": messages})
# Markdown(results['response'][0].content)
# pio.from_json(results['chart_json'])
# pd.DataFrame(results.get("reco_list")).head(10)

# CONCLUSIONS
# - We created a Segment Recommender Agent in the same style as the Customer Segmentation Agent.
# - It recomputes segment→top-K product affinities, applies simple user constraints, recommends one product per lead, and returns a donut chart.
# - This modular agent can be integrated into your supervised team to drive personalized offer selection.
