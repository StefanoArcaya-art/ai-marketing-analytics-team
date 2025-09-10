# BUSINESS SCIENCE UNIVERSITY
# PYTHON FOR GENERATIVE AI COURSE
# MULTI-AGENTS (AGENTIAL SUPERVISION)
# ***

# GOAL: Make a segment-aware recommender AI agent that:
# - Recomputes segment → top-K product affinities from the database
# - Chooses the "next product" for each segment (top-ranked by purchases, tie-break on revenue proxy)
# - Returns a readable report with recommendations by segment + concise affinity notes
# - OPTIONAL: The agent can ALSO return a user-level target list (e.g., "recommend the next product for top 30 users in segment 2")

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
import json
import re

# DATA SCIENCE
import pandas as pd
from sqlalchemy import create_engine

# DEBUGGING
from pprint import pprint
from IPython.display import Markdown, display

# CUSTOM (helper to fetch last human question, if available)
from marketing_analytics_team.agents.utils import get_last_human_message


# --------------------
# SETUP
# --------------------

db_path = "sqlite:///data/database-sql-transactions/leads_scored_segmentation.db"
model = "gpt-4.1-mini"

os.environ["OPENAI_API_KEY"] = yaml.safe_load(open('../credentials.yml'))['openai']
llm = ChatOpenAI(model=model, temperature=0)

# Key Inputs for affinity computation:
LOOKBACK_DAYS = 365
TOPK_PER_SEGMENT = 3


# --------------------
# * 2.1 MAKING THE CUSTOM AGENT - SEGMENT RECOMMENDER
# --------------------

# Recommender Agent Prompt (no bare curly braces that confuse PromptTemplate)
segment_recommender_prompt = PromptTemplate(
    template="""
    You are an expert marketing analyst for Business Science (a premium data-science education platform).
    Based on segment-level product affinities and a proposed deterministic recommendation, generate a concise
    business-facing summary and a markdown table of the recommended "next product" per segment.

    INPUTS:
    - SEGMENT_AFFINITIES: a JSON list of records with fields:
      segment, product_id, product_name, purchase_count, revenue_proxy, rank_in_segment
      Each segment has up to K rows, sorted best-to-worst (rank_in_segment = 1 is strongest).
    - PROPOSED_RECOMMENDATIONS: JSON mapping of segment to an object with fields:
      "product_id", "product_name", and "rationale" that we have preselected.

    YOUR TASK:
    1) Produce a short "General Response" explaining the logic used (top purchases last 12 months, ties by revenue).
    2) Return a markdown table named "Next Product by Segment" with columns:
       segment, product_name, product_id, rationale
    3) Provide a compact "Affinity Notes" section with 1–2 bullets per segment summarizing the top options (K ≤ 3).

    RETURN STRICT JSON with fields:
    {{
        "general_response": "1-2 paragraph summary of the overall recommendation approach.",
        "next_products_table": "Markdown table (segment | product_name | product_id | rationale)",
        "affinity_notes": "Markdown bullets summarizing key affinities by segment (short)."
    }}

    SEGMENT_AFFINITIES: {segment_affinities}
    PROPOSED_RECOMMENDATIONS: {proposed_recommendations}
    """,
    input_variables=["segment_affinities", "proposed_recommendations"]
)

segment_recommender_parser = segment_recommender_prompt | llm | JsonOutputParser()

# Graph State
class GraphState(TypedDict):
    messages: Sequence[BaseMessage]
    response: Sequence[BaseMessage]
    recommendations_by_segment: dict
    affinity_notes: str
    next_products_table: str
    sql_query: str
    # NEW (Option B):
    user_reco_list: list
    user_table_markdown: str


def segment_recommender_node(state: GraphState) -> GraphState:
    print("---SEGMENT RECOMMENDER AGENT---")

    # ---------- SAFETY DEFAULTS (prevents UnboundLocalError) ----------
    user_reco_list: list = []
    user_table_markdown: str = ""
    general_response = ""
    next_products_table = ""
    affinity_notes = ""

    # --------------------
    # 1) Connect to database & load tables
    # --------------------
    engine = create_engine(db_path)
    conn = engine.connect()

    leads_query = """
    SELECT user_email, user_full_name, p1, segment, made_purchase
    FROM leads_scored
    """
    transactions_query = """
    SELECT transaction_id, purchased_at, user_email, product_id
    FROM transactions
    """
    products_query = """
    SELECT product_id, description AS product_name, suggested_price
    FROM products
    """

    df_leads = pd.read_sql(leads_query, conn)
    df_txn   = pd.read_sql(transactions_query, conn)
    df_prod  = pd.read_sql(products_query, conn)
    conn.close()

    # --------------------
    # 2) Lookback filter (robust to mixed dtypes)
    # --------------------
    if "purchased_at" in df_txn.columns:
        df_txn["purchased_at"] = pd.to_datetime(df_txn["purchased_at"], errors="coerce")
        if df_txn["purchased_at"].notna().any():
            cutoff = df_txn["purchased_at"].max() - pd.Timedelta(days=LOOKBACK_DAYS)
            df_txn = df_txn[df_txn["purchased_at"] >= cutoff]

    # --------------------
    # 3) Build segment → top-K product affinities (same logic as Part 1)
    # --------------------
    df_txn_seg = df_txn.merge(df_leads[["user_email", "segment"]], on="user_email", how="left")
    df_txn_seg["segment"] = df_txn_seg["segment"].fillna(0).astype(int)

    df_txn_seg = df_txn_seg.merge(
        df_prod[["product_id", "product_name", "suggested_price"]],
        on="product_id",
        how="left"
    )
    df_txn_seg["product_name"] = df_txn_seg["product_name"].fillna(df_txn_seg["product_id"].astype(str))
    df_txn_seg["suggested_price"] = df_txn_seg["suggested_price"].fillna(0.0)

    agg = (
        df_txn_seg
        .groupby(["segment", "product_id", "product_name"], as_index=False)
        .agg(
            purchase_count=("transaction_id","count"),
            revenue_proxy=("suggested_price","sum")
        )
    )
    agg = agg.sort_values(["segment","purchase_count","revenue_proxy"], ascending=[True,False,False])
    agg["rank_in_segment"] = agg.groupby("segment").cumcount() + 1
    topk = agg[agg["rank_in_segment"] <= TOPK_PER_SEGMENT].copy()

    # Prepare JSON-friendly affinities for the LLM
    affinities_records = topk[["segment","product_id","product_name","purchase_count","revenue_proxy","rank_in_segment"]] \
        .to_dict(orient="records")
    affinities_json = json.dumps(affinities_records)

    # --------------------
    # 4) Deterministic "next product" per segment (rank 1)
    # --------------------
    recommendations_by_segment = {}
    for seg, g in topk.groupby("segment"):
        g = g.sort_values(["rank_in_segment"])
        top_row = g.iloc[0]
        recommendations_by_segment[str(int(seg))] = {
            "product_id": float(top_row["product_id"]) if pd.notna(top_row["product_id"]) else None,
            "product_name": str(top_row["product_name"]),
            "rationale": "Top purchase_count in last 12 months; tie-broken by revenue_proxy."
        }
    recos_json = json.dumps(recommendations_by_segment)

    # --------------------
    # 5) Ask LLM to produce a concise report (using our deterministic picks)
    # --------------------
    try:
        llm_out = segment_recommender_parser.invoke({
            "segment_affinities": affinities_json,
            "proposed_recommendations": recos_json
        })
        general_response = llm_out.get("general_response","")
        next_products_table = llm_out.get("next_products_table","")
        affinity_notes = llm_out.get("affinity_notes","")
    except Exception as e:
        print("LLM formatting warning:", e)
        # Safe fallbacks
        rows = []
        for seg, rec in recommendations_by_segment.items():
            rows.append((seg, rec["product_name"], int(rec["product_id"]) if rec["product_id"] is not None else "NA", rec["rationale"]))
        df_table = pd.DataFrame(rows, columns=["segment","product_name","product_id","rationale"])
        next_products_table = df_table.to_markdown(index=False)
        affinity_notes = "- Affinity notes unavailable due to LLM formatting error."
        general_response = (
            "Recommended the top product per segment based on purchase frequency within the last 12 months "
            "(ties broken by revenue proxy)."
        )

    # --------------------
    # 6) OPTIONAL: Build a user-level target list when the prompt requests it
    #     e.g., "top 30 users in segment 2", supports "non-buyer(s)" and "ignore purchase history"
    # --------------------
    last_msg = get_last_human_message(state.get("messages", []))
    last_text = ((last_msg.content if last_msg else "") or "").lower()

    seg_match = re.search(r"(?:segment|seg)\s*[:#]?\s*(\d+)", last_text)
    # support "top 30", "top-30", "top30", or "<n> users"
    top_match = re.search(r"top\s*[- ]?(\d+)", last_text) or re.search(r"\b(\d+)\s+users?\b", last_text)
    non_buyers_only = bool(re.search(r"non[- ]?buyer", last_text))
    ignore_history = bool(
        re.search(r"ignore\s+(?:purchase|buy)\s+history", last_text)
        or "include previous buyers" in last_text
        or "all users" in last_text
    )

    if seg_match and top_match:
        target_seg = int(seg_match.group(1))
        top_n = int(top_match.group(1))

        # Per-segment ranked products (top-K)
        topk_by_segment = {
            int(seg): g.sort_values("rank_in_segment")[["product_id", "product_name"]]
                        .to_dict(orient="records")
            for seg, g in topk.groupby("segment")
        }
        seg_products = topk_by_segment.get(target_seg, [])
        pid_to_name = {int(p["product_id"]): p["product_name"] for p in seg_products if pd.notna(p["product_id"])}

        # Purchases per user
        purchases_by_user = (
            df_txn.dropna(subset=["user_email"])
                 .dropna(subset=["product_id"])
                 .assign(product_id=lambda d: d["product_id"].astype(int))
                 .groupby("user_email")["product_id"]
                 .apply(set)
                 .to_dict()
        )

        # Candidate pool
        cands = df_leads[df_leads["segment"] == target_seg].copy()
        if non_buyers_only:
            cands = cands[cands["made_purchase"].fillna(0) == 0]
        cands = cands.sort_values("p1", ascending=False).copy()

        def next_eligible_product_for_user(email: str):
            if not seg_products:
                return None, None
            if ignore_history:
                pid = int(seg_products[0]["product_id"]) if pd.notna(seg_products[0]["product_id"]) else None
                return pid, pid_to_name.get(pid)
            already = purchases_by_user.get(email, set())
            for item in seg_products:
                pid = int(item["product_id"]) if pd.notna(item["product_id"]) else None
                if pid is None:
                    continue
                if pid not in already:
                    return pid, item["product_name"]
            return None, None

        for _, r in cands.iterrows():
            if len(user_reco_list) >= top_n:
                break
            pid, pname = next_eligible_product_for_user(r["user_email"])
            if pid is None:
                continue
            user_reco_list.append({
                "user_email": r["user_email"],
                "user_full_name": r.get("user_full_name", ""),
                "p1": float(r["p1"]) if pd.notna(r["p1"]) else 0.0,
                "segment": int(r["segment"]) if pd.notna(r["segment"]) else target_seg,
                "product_id": pid,
                "product_name": pname
            })

        if user_reco_list:
            _df = pd.DataFrame(user_reco_list)[
                ["user_full_name","user_email","p1","segment","product_name","product_id"]
            ]
            user_table_markdown = _df.to_markdown(index=False)

    # --------------------
    # 7) SQL provenance (for audit/debug)
    # --------------------
    sql_used = (
        "SELECT user_email, user_full_name, p1, segment, made_purchase FROM leads_scored; "
        "SELECT transaction_id, purchased_at, user_email, product_id FROM transactions; "
        "SELECT product_id, description AS product_name, suggested_price FROM products;"
    )

    # --------------------
    # 8) Assemble final response
    # --------------------
    formatted_response = (
        f"**Segment Recommender Result**\n\n"
        f"{general_response}\n\n"
        f"### Next Product by Segment\n\n{next_products_table}\n\n"
        f"### Affinity Notes\n\n{affinity_notes}"
    )
    if user_table_markdown:
        formatted_response += (
            f"\n\n### Target List\n"
            f"_Returned by request parsed from your prompt (segment & top-N, plus optional non-buyer / history filters)._"
            f"\n\n_Total users returned: **{len(user_reco_list)}**._\n\n{user_table_markdown}"
        )

    return {
        "response": [AIMessage(content=formatted_response, name="SegmentRecommenderAgent")],
        "recommendations_by_segment": recommendations_by_segment,
        "affinity_notes": affinity_notes,
        "next_products_table": next_products_table,
        "sql_query": sql_used,
        "user_reco_list": user_reco_list,
        "user_table_markdown": user_table_markdown
    }



# Create LangGraph workflow
workflow = StateGraph(GraphState)
workflow.add_node("segment_recommender", segment_recommender_node)
workflow.add_edge(START, "segment_recommender")
workflow.add_edge("segment_recommender", END)

app = workflow.compile()
app


# --------------------
# * 2.2 RUNNING THE RECOMMENDER AGENT — EXAMPLES (Option B in action)
# --------------------

# EXAMPLE A: Segment-level "next product" report (no user list)
messages = [HumanMessage(content="Recommend the next product by segment based on recent affinities.")]

results = app.invoke({"messages": messages})

display(Markdown(results['response'][0].content))

results['recommendations_by_segment']  # raw mapping


# EXAMPLE B: Ask for a user-level list — "top 30 users in segment 2"
# Default behavior: exclude products a user already owns and choose the first eligible from top-K
messages = [HumanMessage(content="Recommend the next product by segment and give me the top 30 users in segment 2.")]

results = app.invoke({"messages": messages})

display(Markdown(results['response'][0].content))

results.get("user_reco_list", [])


# EXAMPLE C: Same as B, but **non-buyers only**
messages = [HumanMessage(content="For segment 2, provide the top 30 users by p1 (non-buyers only) and recommend the next product.")]

results = app.invoke({"messages": messages})

display(Markdown(results['response'][0].content))

len(results.get("user_reco_list", []))


# EXAMPLE D: Force a full Top-N even if most already purchased — **ignore purchase history**
messages = [HumanMessage(content="Segment 2: give me the top 30 users by p1 and recommend the next product, ignore purchase history.")]

results = app.invoke({"messages": messages})

display(Markdown(results['response'][0].content))

len(results.get("user_reco_list", []))
