# BUSINESS SCIENCE UNIVERSITY
# PYTHON FOR GENERATIVE AI COURSE
# MULTI-AGENTS (AGENTIAL SUPERVISION)
# ***

# GOAL: Make a segment-aware recommender AI agent that:
# - Recomputes segment → top-K product affinities from the database
# - Chooses the "next product" for each segment (top-ranked by purchases, tie-break on revenue proxy)
# - Returns a readable report with recommendations by segment + concise affinity notes
# - Option B: The agent can RETURN results (e.g., user-level target list) based on a natural-language request

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

# --------------------
# SETUP
# --------------------

db_path = "sqlite:///data/database-sql-transactions/leads_scored_segmentation.db"
model = "gpt-4.1-mini"

os.environ["OPENAI_API_KEY"] = yaml.safe_load(open('../credentials.yml'))['openai']
llm = ChatOpenAI(model=model, temperature=0)

# Affinity computation knobs
LOOKBACK_DAYS = 365
TOPK_PER_SEGMENT = 3

# --------------------
# * 2.1 TEST THE LLM (optional)
# --------------------
_ = llm.invoke("Ready.")  # sanity check


# --------------------
# * 2.2 SEGMENT RECOMMENDER 
# --------------------

# Helper: get last human message content (self-contained)
def get_last_human_message(messages: Sequence[BaseMessage]) -> str:
    if not messages:
        return ""
    for m in reversed(messages):
        if isinstance(m, HumanMessage) or getattr(m, "type", "") == "human":
            return m.content or ""
    return ""

# Helper: parse user intent like “top 30 users in segment 2”, with optional flags
def parse_request(text: str):
    if not text:
        return {}
    text_l = text.lower()

    seg_match = re.search(r"(?:segment|seg)\s*[:=]?\s*(\d+)", text_l)
    top_match = re.search(r"(?:top|first)\s+(\d+)", text_l)
    nonbuyers = bool(re.search(r"non[-\s]?buyers?|exclude\s+buyers", text_l))
    ignore_hist = bool(re.search(r"ignore\s+(?:purchase|buy) history|regardless\s+of\s+(?:purchases|history)", text_l))

    return {
        "segment_id": int(seg_match.group(1)) if seg_match else None,
        "top_n": int(top_match.group(1)) if top_match else None,
        "non_buyers_only": nonbuyers,
        "ignore_history": ignore_hist
    }


def make_segment_recommender_agent(model="gpt-4.1-mini", db_path=db_path):
    """Builds a LangGraph workflow with a segment recommender agent that returns results."""
    
    if isinstance(model, str):
        llm = ChatOpenAI(model=model, temperature=0)
    else:
        llm = model
    
    # Prompt (escape literal braces with double {{ }})
    segment_recommender_prompt = PromptTemplate(
        template="""
    You are an expert marketing analyst for Business Science (a premium data-science education platform).
    Based on segment-level product affinities and deterministic picks, generate a concise business-facing
    summary and a markdown table of the recommended "next product" per segment.

    INPUTS:
    - SEGMENT_AFFINITIES: JSON list of records with fields:
    segment, product_id, product_name, purchase_count, revenue_proxy, rank_in_segment
    (Each segment has up to K rows, sorted best-to-worst; rank_in_segment = 1 is strongest.)
    - PROPOSED_RECOMMENDATIONS: JSON mapping of segment → {{ "product_id", "product_name", "rationale" }}.

    YOUR TASK:
    1) Produce a short "General Response" describing the approach (top purchases last 12 months; tie by revenue).
    2) Return a markdown table "Next Product by Segment" with columns:
    segment | product_name | product_id | rationale
    3) Provide a compact "Affinity Notes" section with 1–2 bullets per segment summarizing the top options (K ≤ 3).

    RETURN STRICT JSON with fields:
    {{
    "general_response": "1-2 paragraph summary of the recommendation approach.",
    "next_products_table": "Markdown table (segment | product_name | product_id | rationale)",
    "affinity_notes": "Markdown bullets per segment (short)."
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
        # Primary outputs
        recommendations_by_segment: dict
        affinity_notes: str
        next_products_table: str
        sql_query: str
        # Option B: user-level list outputs
        user_reco_list: list
        user_table_markdown: str


    def segment_recommender_node(state: GraphState) -> GraphState:
        print("---SEGMENT RECOMMENDER AGENT---")

        # --------------------
        # 1) Connect to DB & load tables
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
        # 2) Lookback filter (robust to tz-aware/tz-naive)
        # --------------------
        if "purchased_at" in df_txn.columns:
            df_txn["purchased_at"] = pd.to_datetime(df_txn["purchased_at"], errors="coerce")
            # Drop timezone to avoid comparison errors
            if getattr(df_txn["purchased_at"].dt, "tz", None) is not None:
                df_txn["purchased_at"] = df_txn["purchased_at"].dt.tz_convert(None)
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

        # Affinities JSON for the LLM
        affinities_json = topk[["segment","product_id","product_name","purchase_count","revenue_proxy","rank_in_segment"]].to_json(orient="records")

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
        # 5) Ask LLM to produce report (with our deterministic picks)
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
                pid_val = int(rec["product_id"]) if rec["product_id"] is not None else "NA"
                rows.append((seg, rec["product_name"], pid_val, rec["rationale"]))
            df_table = pd.DataFrame(rows, columns=["segment","product_name","product_id","rationale"])
            next_products_table = df_table.to_markdown(index=False)
            affinity_notes = "- Affinity notes unavailable due to LLM formatting error."
            general_response = (
                "Recommended the top product per segment based on purchase frequency within the last 12 months "
                "(ties broken by revenue proxy)."
            )

        # --------------------
        # 6) Option B: User-level target list requested?
        # --------------------
        last_question = get_last_human_message(state.get("messages", []))
        req = parse_request(last_question)
        user_reco_list = []
        user_table_markdown = ""  # initialize to avoid UnboundLocalError

        if req.get("segment_id") is not None and req.get("top_n") is not None:
            seg_id = int(req["segment_id"])
            top_n = int(req["top_n"])
            non_buyers_only = bool(req.get("non_buyers_only", False))
            ignore_history = bool(req.get("ignore_history", False))

            # Recommended product for this segment
            reco = recommendations_by_segment.get(str(seg_id))
            if reco is not None:
                next_pid = reco["product_id"]
                next_pname = reco["product_name"]

                # Filter leads in this segment
                candidates = df_leads[df_leads["segment"] == seg_id].copy()

                # Non-buyers only?
                if non_buyers_only and "made_purchase" in candidates.columns:
                    candidates["made_purchase"] = candidates["made_purchase"].fillna(0)
                    candidates = candidates[candidates["made_purchase"] == 0]

                # Exclude users who already bought the recommended product (unless ignore_history)
                if not ignore_history and next_pid is not None and pd.notna(next_pid):
                    bought_next = (
                        df_txn[df_txn["product_id"] == next_pid]
                        .groupby("user_email").size().reset_index(name="cnt")
                    )
                    candidates = candidates.merge(bought_next, on="user_email", how="left")
                    candidates = candidates[candidates["cnt"].fillna(0) == 0]
                    candidates = candidates.drop(columns=["cnt"], errors="ignore")

                # Rank by p1 and take top-N
                candidates = candidates.sort_values("p1", ascending=False).head(top_n).copy()
                candidates["product_id"] = int(next_pid) if pd.notna(next_pid) else None
                candidates["product_name"] = next_pname

                # Build user list + markdown table
                user_reco_list = candidates[["user_email","user_full_name","p1","segment","product_id","product_name"]] \
                    .to_dict(orient="records")
                user_table_markdown = candidates[["user_full_name","user_email","p1","segment","product_name","product_id"]].to_markdown(index=False)

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
                f"_Returned by request parsed from your prompt (segment & top-N, plus optional non-buyer/ignore-history flags)._"
                f"\n\n{user_table_markdown}"
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


    # Build workflow
    workflow = StateGraph(GraphState)
    workflow.add_node("segment_recommender", segment_recommender_node)
    workflow.add_edge(START, "segment_recommender")
    workflow.add_edge("segment_recommender", END)

    app = workflow.compile()
    
    return app

# Make the agent
app = make_segment_recommender_agent(model=model, db_path=db_path)


# --------------------
# * 2.3 RUNNING THE RECOMMENDER AGENT (EXAMPLES)
# --------------------

# Example A: General recommendation summary
messages = [HumanMessage(content="Recommend the next product by segment based on recent affinities.")]
results = app.invoke({"messages": messages})

# Display the AI message response (includes the recommendations table + affinity notes)
display(Markdown(results['response'][0].content))
# View the raw mapping if desired
results['recommendations_by_segment']


# Example B: Ask for a user-level list — "top 30 users in segment 2"
messages = [HumanMessage(content="Recommend the next product by segment and give me the top 30 users in segment 2.")]
results = app.invoke({"messages": messages})

display(Markdown(results['response'][0].content))
# Programmatic access to the user list
results["user_reco_list"]