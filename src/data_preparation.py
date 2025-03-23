import json

def prepare_data(transactions):
    extracted_data = []
    for txn in transactions.get('transactions', []):
        extracted_data.append({
            "merchant": txn.get("name"),
            "amount": txn.get("amount"),
            "date": txn.get("date"),
            "category": " > ".join(txn.get("category", [])),
            "account_id": txn.get("account_id")
        })
    return extracted_data
