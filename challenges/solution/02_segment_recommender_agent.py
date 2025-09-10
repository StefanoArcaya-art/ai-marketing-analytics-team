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

# Helper: Parse things like: "top 30 users in segment 2" (optionally: "non-buyers")
def parse_request(text: str):
    if not text:
        return {}
    text_l = text.lower()

    seg_match = re.search(r"(?:segment|seg)\s*[:=]?\s*(\d+)", text_l)
    top_match = re.search(r"(?:top|first)\s+(\d+)", text_l)
    nonbuyers = bool(re.search(r"non[-\s]?buyers?|exclude\s+buyers", text_l))

    return {
        "segment_id": int(seg_match.group(1)) if seg_match else None,
        "top_n": int(top_match.group(1)) if top_match else None,
        "non_buyers_only": nonbuyers
    }

def make_segment_recommender_agent(model: str = "gpt-4.1-mini", db_path: str = db_path):
    """Build the segment recommender agent as a LangGraph workflow."""
    
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
        # User-level outputs
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
        df_txn_all = pd.read_sql(transactions_query, conn)   # keep full history for ownership checks
        df_prod  = pd.read_sql(products_query, conn)
        conn.close()

        # --------------------
        # 2) Lookback filter for AFFINITIES only (keep df_txn_all for ownership)
        # --------------------
        df_txn = df_txn_all.copy()
        if "purchased_at" in df_txn.columns:
            df_txn["purchased_at"] = pd.to_datetime(df_txn["purchased_at"], errors="coerce")
            if getattr(df_txn["purchased_at"].dt, "tz", None) is not None:
                df_txn["purchased_at"] = df_txn["purchased_at"].dt.tz_convert(None)
            cutoff = df_txn["purchased_at"].max() - pd.Timedelta(days=LOOKBACK_DAYS)
            df_txn = df_txn[df_txn["purchased_at"] >= cutoff]

        # --------------------
        # 3) Build segment → top-K product affinities (last 12 months)
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

        # LLM affinities payload
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
        # 5) Ask LLM for segment-level report
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
        # 6) USER-LEVEL PERSONALIZED LIST (top-N in a segment)
        #    Logic: For each user, recommend FIRST product they don't own among the segment's top-3.
        #    If owns all top-3 → no recommendation for that user.
        # --------------------
        last_question = get_last_human_message(state.get("messages", []))
        req = parse_request(last_question)

        user_reco_list = []
        user_table_markdown = ""  # ensure defined for final formatting

        if req.get("segment_id") is not None and req.get("top_n") is not None:
            seg_id = int(req["segment_id"])
            top_n = int(req["top_n"])
            non_buyers_only = bool(req.get("non_buyers_only", False))

            # Build segment → top-3 product list (ints) for personalization
            seg_top3_map = {}
            for s, g in topk.groupby("segment"):
                g2 = g.sort_values("rank_in_segment")
                pids = [int(pid) for pid in list(g2["product_id"].dropna().astype(int))[:TOPK_PER_SEGMENT]]
                seg_top3_map[int(s)] = pids

            # Product name map (int → name)
            prod_name_map = {
                int(pid): name
                for pid, name in df_prod[["product_id","product_name"]].dropna().assign(product_id=lambda d: d["product_id"].astype(int)).values
            }

            # Ownership map (user_email → set(int product_ids))
            owned_map = (
                df_txn_all.dropna(subset=["user_email", "product_id"])
                        .assign(product_id=lambda d: d["product_id"].astype(int))
                        .groupby("user_email")["product_id"]
                        .apply(lambda s: set(s.tolist()))
                        .to_dict()
            )

            # Candidates in the requested segment
            candidates = df_leads[df_leads["segment"] == seg_id].copy()

            # Optional filter: non-buyers overall (not required by default)
            if non_buyers_only and "made_purchase" in candidates.columns:
                candidates["made_purchase"] = candidates["made_purchase"].fillna(0)
                candidates = candidates[candidates["made_purchase"] == 0]

            # Rank by p1 and take top-N (do NOT exclude based on ownership)
            candidates = candidates.sort_values("p1", ascending=False).head(top_n).copy()

            # Personalize recommendation per user using segment top-3
            seg_top3 = seg_top3_map.get(seg_id, [])
            def pick_next_for_user(email: str) -> tuple[int | None, str | None, str]:
                owned = owned_map.get(email, set())
                for pid in seg_top3:
                    if pid not in owned:
                        return pid, prod_name_map.get(pid), "recommended"
                # owns all top-3 or no top-3 available
                if not seg_top3:
                    return None, None, "no_affinity_data"
                return None, None, "owns_all_top3"

            rec_pids = []
            rec_names = []
            rec_status = []
            for _, row in candidates.iterrows():
                pid, name, status = pick_next_for_user(row["user_email"])
                rec_pids.append(pid)
                rec_names.append(name)
                rec_status.append(status)

            candidates["recommended_product_id"] = [int(p) if p is not None and not (isinstance(p, float) and math.isnan(p)) else None for p in rec_pids]
            candidates["recommended_product_name"] = rec_names
            candidates["recommendation_status"] = rec_status

            # Build outputs
            user_reco_list = candidates[[
                "user_email","user_full_name","p1","segment",
                "recommended_product_id","recommended_product_name","recommendation_status"
            ]].to_dict(orient="records")

            # Pretty table
            df_disp = candidates[[
                "user_full_name","user_email","p1","segment",
                "recommended_product_name","recommended_product_id","recommendation_status"
            ]].copy()
            user_table_markdown = df_disp.to_markdown(index=False)

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
                f"\n\n### Target List (Personalized using segment top-3)\n"
                f"_For each user: recommend the first product they don't already own among the segment's top-3; "
                f"if they own all top-3, no recommendation is made._\n\n"
                f"{user_table_markdown}"
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
app

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