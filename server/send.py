import pandas as pd
from pymongo import MongoClient

# ===== CONFIG =====
CSV_FILE = "synthetic_credit_dataset_80features (1).csv"          # your CSV file name (same directory)
MONGO_URI = "mongodb://127.0.0.1:27017/"
DB_NAME = "cpuHackathon"
COLLECTION_NAME = "csv"

# ===== CONNECT TO MONGODB =====
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# ===== READ CSV =====
df = pd.read_csv(CSV_FILE)

# ===== INSERT ROW BY ROW WITH SERIAL NUMBER =====
start_index = collection.count_documents({}) + 1  # continue numbering if collection already has data

for i, row in df.iterrows():
    serial_number = f"EN-{start_index + i:06d}"  # EN-000001 format

    doc = row.to_dict()
    doc["entry_id"] = serial_number  # add serial number field

    # Insert into MongoDB
    collection.insert_one(doc)

    print(f"Inserted: {serial_number}")

print("✅ All records inserted successfully!")