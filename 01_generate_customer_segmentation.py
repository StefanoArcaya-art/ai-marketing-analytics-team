# BUSINESS SCIENCE UNIVERSITY
# PYTHON FOR GENERATIVE AI COURSE
# MULTI-AGENTS (AGENTIAL SUPERVISION)
# ***

# GOAL: Make a customer segmentation AI agent based similar to the approach used in Clinic #2 for the Lead Scoring (Business Intelligence Expert).


# * PART 1: MACHINE LEARNING SEGMENTATION
# This script generates customer segments based on transaction data and updates the leads_scored table in the SQLite database.

# LIBRARIES
import pandas as pd
import sqlalchemy as sql
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

N_CLUSTERS = 5

PATH_CRM_DATABASE = "sqlite:///data/database-sql-transactions/leads_scored_segmentation.db"

# Connect to SQL database using SQLAlchemy

engine = sql.create_engine(PATH_CRM_DATABASE)
conn = engine.connect()

# Load data
transactions = pd.read_sql("SELECT * FROM transactions", conn)
transactions

leads_scored = pd.read_sql("SELECT * FROM leads_scored", conn)
leads_scored

# Calculate purchase frequency
purchase_freq = transactions.groupby("user_email").size().reset_index(name="purchase_frequency")

# Merge features
customer_features = leads_scored[["user_email", "p1", "member_rating"]].merge(
    purchase_freq, on="user_email", how="left"
)

# Fill missing purchase frequency with 0 (for users with no transactions)
customer_features["purchase_frequency"] = customer_features["purchase_frequency"].fillna(0)

# Handle any missing values in member_rating or p1
customer_features["member_rating"] = customer_features["member_rating"].fillna(customer_features["member_rating"].mean())

customer_features["p1"] = customer_features["p1"].fillna(customer_features["p1"].mean())

customer_features

# Standardize features
features = ["purchase_frequency", "p1", "member_rating"]
X = customer_features[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42)
customer_features["segment"] = kmeans.fit_predict(X_scaled)

# Drop any column that starts with "segment"
to_drop = leads_scored.filter(regex="^segment").columns
leads_scored = leads_scored.drop(columns=to_drop, errors="ignore")

# Merge new segments into leads_scored
leads_scored["segment"] = customer_features["segment"]
leads_scored

# Fill any missing segments with 0 (default segment)
leads_scored["segment"] = leads_scored["segment"].fillna(0).astype(int)

# Save updated leads_scored back to database
leads_scored.to_sql("leads_scored", conn, if_exists="replace", index=False)

# Finalize the SQL insertion
conn.commit()

# Verify update
pd.read_sql("SELECT * FROM leads_scored", conn)

# Close connection
conn.close()

# * CONCLUSIONS
# - We have NOT done any AI. 
# - We have created customer segments using K-Means clustering based on purchase frequency, lead score (p1), and engagement rating (member_rating).
# - The next step is to create an AI agent that can analyze these segments and generate descriptive labels and insights.

