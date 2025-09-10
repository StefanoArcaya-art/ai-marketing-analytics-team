# BUSINESS SCIENCE UNIVERSITY
# PYTHON FOR GENERATIVE AI COURSE
# MULTI-AGENTS (AGENTIAL SUPERVISION)
# ***
# GOAL (Part 1): Build segment → top-K product affinities from transactions.

# LIBRARIES
import pandas as pd
from sqlalchemy import create_engine
from IPython.display import Markdown, display

# CONFIG
DB_PATH = "sqlite:///data/database-sql-transactions/leads_scored_segmentation.db"
LOOKBACK_DAYS = 365            # purchases in the last 12 months
TOPK_PER_SEGMENT = 3           # keep it small/simple
DEFAULT_PRICE_IF_MISSING = 0.0

# 1) Load tables
engine = create_engine(DB_PATH)
with engine.connect() as conn:
    # NOTE: we expect segments to be in leads_scored (from your segmentation step)
    df_leads = pd.read_sql("SELECT * FROM leads_scored", conn)   # must have user_email, p1, segment (int), optionally made_purchase
    df_txn   = pd.read_sql("SELECT * FROM transactions", conn)   # transaction_id, purchased_at, user_email, product_id
    df_prod  = pd.read_sql("SELECT * FROM products", conn)       # product_id, description, suggested_price

# 2) Filter transactions by lookback window (if timestamp exists)
cutoff = df_txn["purchased_at"].max() - pd.Timedelta(days=LOOKBACK_DAYS)
df_txn = df_txn[df_txn["purchased_at"] >= cutoff]

# 3) Join transactions → segment (via email)
df_txn_seg = df_txn.merge(df_leads[["user_email", "segment"]], on="user_email", how="left")
df_txn_seg["segment"] = df_txn_seg["segment"].fillna(0).astype(int)

# 4) Join products to get names/prices
df_txn_seg = df_txn_seg.merge(
    df_prod[["product_id", "description", "suggested_price"]],
    on="product_id",
    how="left"
)
df_txn_seg["suggested_price"] = df_txn_seg["suggested_price"].fillna(DEFAULT_PRICE_IF_MISSING)
df_txn_seg["description"] = df_txn_seg["description"].fillna(df_txn_seg["product_id"].astype(str))

# 5) Build segment × product affinities
#    - count purchases
#    - revenue proxy = count × suggested_price
agg = (
    df_txn_seg
    .groupby(["segment", "product_id", "description"], as_index=False)
    .agg(purchase_count=("transaction_id", "count"),
         revenue_proxy=("suggested_price", "sum"))   # one row per txn; sum suggested_price approximates 1-unit purchases
)
# Rank within segment by purchase_count then revenue_proxy
agg["rank_in_segment"] = agg.sort_values(
    ["segment", "purchase_count", "revenue_proxy"],
    ascending=[True, False, False]
).groupby("segment").cumcount() + 1

# Keep top-K per segment
topk = agg[agg["rank_in_segment"] <= TOPK_PER_SEGMENT].copy()

# 6) Create a mapping: segment -> list of top-K products (id, name, price)
# Derive avg price from aggregates (no 'suggested_price' column in topk)
segment_to_products = (
    topk.sort_values(["segment", "rank_in_segment"])
        .groupby("segment")
        .apply(lambda g: [
            {
                "product_id": float(pid) if pd.notna(pid) else None,
                "product_name": desc,
                # avg price per txn; protects against division by 0
                "suggested_price": float(rev) / float(cnt) if float(cnt) > 0 else DEFAULT_PRICE_IF_MISSING
            }
            for pid, desc, rev, cnt in zip(
                g["product_id"], g["description"], g["revenue_proxy"], g["purchase_count"]
            )
        ])
        .to_dict()
)


# 7) Report: Top products affinity per segment
from IPython.display import Markdown, display
import pandas as pd
import math

display(Markdown("### Segment → Top-K Product Affinities"))

for seg, prods in segment_to_products.items():
    lines = []
    for p in prods:
        pid = p.get("product_id", None)
        price = p.get("suggested_price", 0.0)

        # product id → "NA" or integer-like string
        if pid is None or (isinstance(pid, float) and (math.isnan(pid))):
            pid_str = "NA"
        else:
            pid_str = f"{int(float(pid))}"

        # price → numeric string with 2 decimals
        try:
            price_val = float(price)
        except (TypeError, ValueError):
            price_val = 0.0
        price_str = f"${price_val:,.2f}"

        name = p.get("product_name", "(unknown)")
        lines.append(f"- **{name}** (ID={pid_str}, {price_str})")

    display(Markdown(f"**Segment {seg}**\n\n" + "\n".join(lines)))

