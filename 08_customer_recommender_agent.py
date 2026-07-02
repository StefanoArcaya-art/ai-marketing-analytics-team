# BUSINESS SCIENCE UNIVERSITY
# PYTHON FOR GENERATIVE AI COURSE
# CUSTOMER RECOMMENDER AGENT (BUILDING ON CUSTOMER SEGMENTATION)

# GOAL:
# Recommend the best-fit products for each customer segment using prior purchases,
# segment propensity scores, and product prices.

# LIBRARIES
from typing import Dict, Sequence, TypedDict, Any
import os
import yaml
import pandas as pd
from sqlalchemy import create_engine

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, START, END

from marketing_analytics_team.agents.utils import get_last_human_message

# SETUP
DB_PATH = "sqlite:///data/database-sql-transactions/leads_scored_segmentation.db"
MODEL = "gpt-4.1-mini"

# Load API key (expects credentials.yml in repo root)
os.environ["OPENAI_API_KEY"] = yaml.safe_load(open("credentials.yml"))["openai"]


# --- Helper functions -------------------------------------------------------------------------

def _label_segment(row: pd.Series) -> str:
    """Lightweight heuristic label so the LLM has a head start."""
    if row["avg_p1"] >= 0.7 and row["avg_purchase_frequency"] >= 1:
        return "High-Value Repeaters"
    if row["avg_p1"] >= 0.6:
        return "Hot Prospects"
    if row["avg_member_rating"] >= 4:
        return "Engaged Nurture"
    if row["avg_p1"] <= 0.35:
        return "Cold / Nurture"
    return "Mid-Funnel Opportunities"


def _build_dataframes(engine) -> Dict[str, pd.DataFrame]:
    """Read and prepare the core tables used in the recommender."""
    df_leads = pd.read_sql(
        "SELECT user_email, segment, p1, member_rating FROM leads_scored", engine
    )
    df_transactions = pd.read_sql(
        "SELECT user_email, product_id, purchased_at FROM transactions", engine
    )
    df_products = pd.read_sql(
        "SELECT product_id, description, suggested_price FROM products", engine
    )
    return {
        "leads": df_leads,
        "transactions": df_transactions,
        "products": df_products,
    }


def _prepare_segment_summary(df_leads: pd.DataFrame, df_transactions: pd.DataFrame) -> pd.DataFrame:
    """Compute segment-level engagement and propensity summary."""
    purchase_freq = (
        df_transactions.groupby("user_email").size().reset_index(name="purchase_frequency")
    )
    df_analysis = df_leads.merge(purchase_freq, on="user_email", how="left")
    df_analysis["purchase_frequency"] = df_analysis["purchase_frequency"].fillna(0)

    df_summary = (
        df_analysis.groupby("segment")
        .agg(
            avg_p1=("p1", "mean"),
            avg_member_rating=("member_rating", "mean"),
            avg_purchase_frequency=("purchase_frequency", "mean"),
            customer_count=("user_email", "count"),
        )
        .reset_index()
    )
    df_summary["avg_p1"] = df_summary["avg_p1"].round(3)
    df_summary["avg_member_rating"] = df_summary["avg_member_rating"].round(2)
    df_summary["avg_purchase_frequency"] = df_summary["avg_purchase_frequency"].round(2)
    df_summary["segment_label"] = df_summary.apply(_label_segment, axis=1)
    return df_summary


def _prepare_product_performance(
    df_transactions: pd.DataFrame, df_leads: pd.DataFrame, df_products: pd.DataFrame
) -> pd.DataFrame:
    """Aggregate product demand by segment."""
    df_seg_txn = df_transactions.merge(df_leads[["user_email", "segment"]], on="user_email", how="left")
    df_seg_txn = df_seg_txn.merge(df_products, on="product_id", how="left")

    if df_seg_txn.empty:
        return pd.DataFrame(columns=["segment", "product_id", "description", "suggested_price", "purchase_count", "revenue"])

    df_seg_txn["suggested_price"] = df_seg_txn["suggested_price"].fillna(0)
    df_seg_txn["revenue"] = df_seg_txn["suggested_price"]

    df_product_perf = (
        df_seg_txn.groupby(["segment", "product_id", "description", "suggested_price"])
        .agg(
            purchase_count=("product_id", "size"),
            revenue=("revenue", "sum"),
        )
        .reset_index()
    )
    df_product_perf["revenue"] = df_product_perf["revenue"].round(2)
    return df_product_perf


def make_customer_recommender_agent(model: Any = MODEL, db_path: str = DB_PATH, temperature: float = 0):
    """
    Create an agent that recommends products per customer segment.

    Returns
    -------
    app : Compiled LangGraph workflow.
    """
    # Handle model initialization
    if isinstance(model, str):
        llm = ChatOpenAI(model=model)
    else:
        llm = model
    llm.temperature = temperature

    recommendation_prompt = PromptTemplate(
        template="""
        You are a marketing analytics recommender. Pair each customer segment with products
        that fit their behavior and value. Use the data provided; avoid inventing prices.

        USER QUESTION: {initial_question}
        CHAT HISTORY: {chat_history}
        SEGMENT SUMMARY (with heuristic labels): {segment_summary}
        TOP PRODUCTS BY SEGMENT (purchase counts and revenue): {top_products_by_segment}

        Guidelines:
        - Recommend 1-3 products for each segment, prioritizing higher purchase_count and revenue.
        - When a segment lacks purchases, suggest a logical starter product based on segment propensity.
        - Include price-based guidance (e.g., anchor price, bundle/upsell ideas).
        - Keep the response concise and tactical.

        Return a valid JSON object with:
        {{
            "general_response": "Overall strategy and quick wins",
            "per_segment": [
                {{
                    "segment": "0",
                    "segment_label": "High-Value Repeaters",
                    "recommended_products": [
                        {{
                            "product_id": 101,
                            "product_name": "Course or Bundle Name",
                            "price": 297.0,
                            "reason": "Why it fits this segment"
                        }}
                    ],
                    "campaign_idea": "Short tactic tailored to this segment"
                }}
            ],
            "markdown_table": "Markdown table with columns: segment, segment_label, product_id, product_name, price, reason"
        }}
        """,
        input_variables=["initial_question", "chat_history", "segment_summary", "top_products_by_segment"],
    )

    recommendation_chain = recommendation_prompt | llm | JsonOutputParser()

    class GraphState(TypedDict):
        messages: Sequence[BaseMessage]
        response: Sequence[BaseMessage]
        recommendations_table: str
        per_segment: list
        general_response: str

    def recommendation_node(state: GraphState) -> GraphState:
        print("---CUSTOMER RECOMMENDER AGENT---")
        engine = create_engine(db_path)

        dfs = _build_dataframes(engine)
        df_summary = _prepare_segment_summary(dfs["leads"], dfs["transactions"])
        df_product_perf = _prepare_product_performance(
            dfs["transactions"], dfs["leads"], dfs["products"]
        )

        # Select top 3 products per segment by purchase count then revenue
        df_top_products = (
            df_product_perf.sort_values(["segment", "purchase_count", "revenue"], ascending=[True, False, False])
            .groupby("segment")
            .head(3)
        )

        segment_summary_json = df_summary.to_json(orient="records")
        top_products_json = df_top_products.to_json(orient="records")

        messages = state.get("messages")
        last_question = get_last_human_message(messages)
        last_question = last_question.content if last_question else ""

        result = recommendation_chain.invoke(
            {
                "initial_question": last_question,
                "chat_history": messages,
                "segment_summary": segment_summary_json,
                "top_products_by_segment": top_products_json,
            }
        )
        print("LLM Result Keys:", result.keys())

        # Compose a human-friendly response
        general_response = result.get("general_response", "")
        per_segment = result.get("per_segment", [])
        table_md = result.get("markdown_table", "")

        per_segment_lines = []
        for seg in per_segment:
            seg_id = seg.get("segment", "N/A")
            label = seg.get("segment_label", "")
            campaign = seg.get("campaign_idea", "")
            header = f"- Segment {seg_id} ({label})"
            recs = seg.get("recommended_products", [])
            if recs:
                rec_lines = [
                    f"  * {r.get('product_name', 'Product')} (${r.get('price', 'n/a')}): {r.get('reason', '')}"
                    for r in recs
                ]
                if campaign:
                    rec_lines.append(f"  * Campaign idea: {campaign}")
                per_segment_lines.append("\n".join([header] + rec_lines))
            else:
                per_segment_lines.append(f"{header} — No purchase history; suggest starter product.")

        formatted_response = (
            f"**Overall Strategy**: {general_response}\n\n"
            f"**Segment Recommendations**:\n" + "\n".join(per_segment_lines) + "\n\n"
            f"**Recommendation Table**:\n{table_md}"
        )

        return {
            "response": [AIMessage(content=formatted_response, name="Customer_Recommender")],
            "recommendations_table": table_md,
            "per_segment": per_segment,
            "general_response": general_response,
        }

    workflow = StateGraph(GraphState)
    workflow.add_node("recommender", recommendation_node)
    workflow.add_edge(START, "recommender")
    workflow.add_edge("recommender", END)
    app = workflow.compile()

    return app


# --- Example usage ---------------------------------------------------------------------------
if __name__ == "__main__":
    from IPython.display import Markdown

    agent = make_customer_recommender_agent(MODEL, DB_PATH, temperature=0.4)

    messages = [
        HumanMessage(content="Which products should we push to each segment this month? Focus on revenue impact.")
    ]

    results = agent.invoke({"messages": messages})

    print(results.keys())
    Markdown(results["response"][0].content)
    print(results["recommendations_table"])
