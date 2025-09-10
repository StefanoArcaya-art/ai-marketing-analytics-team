# BUSINESS SCIENCE UNIVERSITY
# PYTHON FOR GENERATIVE AI COURSE
# BUILD YOUR OWN DATA SCIENCE AGENT (STREAMLIT CHALLENGE)
# ***

# CHALLENGE 4: BUILD YOUR OWN DATA SCIENCE AGENT

# DIFFICULTY: INTERMEDIATE

# SPECIFIC ACTIONS:
#  1. Implement make_[INSERT NAME]_agent(model, db_url, **kwargs). 
#     Example: make_recommender_agent()
#
#  2. Your agent should handle natural-language queries and select 
#     from a set of data science tasks such as:
#       • Time-series analysis/forecast (monthly sales, Prophet, etc.)
#       • Recommender (e.g., segment-aware next-product using top-K affinities)
#
#  3. Return BOTH:
#       a) human-readable: AIMessage(content=markdown) in results['response'][0].content
#       b) machine-readable: dicts/tables in results['artifacts'] 
#          (e.g., 'tables', 'fig_json', 'recommendations', 'user_list')
#
#  4. Integrate into a Streamlit app where the user can chat with the agent. 
#     Your solution can be flexible — experiment with additional features or tasks. 
#
# NOTE: A solution (Part 1 & Part 2) will be provided, but the goal 
# is for you to design and implement your own data science agent.

# SAMPLE QUESTIONS:
#  - For product_id 41, compute monthly sales for the last 12 months and plot a time series. 
#    Forecast the next 3 months.
#  - Recommend the next product by segment based on recent affinities (top 3 options). 
#    Give me the top 30 users in segment 2.
#  - Give me non-buyers in segment 3 who have not bought the recommended product, 
#    top 50 by p1.
