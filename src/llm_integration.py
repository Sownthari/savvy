import re
from dateparser import parse
from datetime import datetime, timedelta

CATEGORY_MAP = [
    "food", "groceries", "shopping", "travel", "entertainment",
    "utilities", "health", "education", "insurance", "rent", "income"
]

def extract_date_range(query):
    today = datetime.now()

    if "last month" in query:
        start_date = (today.replace(day=1) - timedelta(days=1)).replace(day=1)
        end_date = today.replace(day=1) - timedelta(days=1)
    elif "this month" in query:
        start_date = today.replace(day=1)
        end_date = today
    elif match := re.search(r"last (\d+) (day|week|month|year)s?", query):
        value, unit = int(match.group(1)), match.group(2)
        if unit == "day":
            start_date = today - timedelta(days=value)
        elif unit == "week":
            start_date = today - timedelta(weeks=value)
        elif unit == "month":
            start_date = (today.replace(day=1) - timedelta(days=1)).replace(day=1) - timedelta(days=30 * (value - 1))
        elif unit == "year":
            start_date = today.replace(year=today.year - value, month=1, day=1)
        end_date = today
    else:
        start_date = None
        end_date = today

    return start_date, end_date

def extract_categories(query):
    detected_categories = [
        category for category in CATEGORY_MAP
        if re.search(rf"\b{category}\b", query, re.IGNORECASE)
    ]

    if "all expenses" in query or "all transactions" in query:
        return CATEGORY_MAP
    if "spending" in query or "expenses" in query:
        return detected_categories if detected_categories else CATEGORY_MAP

    return detected_categories if detected_categories else CATEGORY_MAP

def search_transactions(query, index, data, model):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k=3)
    matched_data = [data[i] for i in indices[0]]
    return matched_data

def generate_response(query, data, index, model, llama3_client):
    matched_data = search_transactions(query, index, data, model)

    # Filter transactions by date range and categories
    start_date, end_date = extract_date_range(query)
    selected_categories = extract_categories(query)

    filtered_data = [
        item for item in matched_data
        if (not start_date or start_date <= datetime.strptime(item['date'], "%Y-%m-%d") <= end_date)
        and item['category'].lower() in selected_categories
    ]

    # Build context for prompt
    context = "\n".join([
        f"{item['date']} - {item['merchant']} - {item['amount']} USD - {item['category']}"
        for item in filtered_data
    ])

    # LLM Prompt Design
    prompt = f"""
    You are a financial assistant specializing in analyzing user transactions.

    ### Transaction Data ###
    {context}

    ### Task ###
    - Identify all expenses categorized as {', '.join(selected_categories)}.
    - Summarize total expenses for the specified period.
    - Provide insights on spending trends if possible.

    Question: {query}
    """

    # Send to LLaMA for answer generation
    response = llama3_client.generate(prompt)
    return response
