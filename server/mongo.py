"""
MongoDB Connection Test Script
Run this file to test if MongoDB is working properly
"""

from pymongo import MongoClient
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DATABASE_NAME = "app"  # Same as your app's database name
COLLECTION_NAME = "test_collection"

def test_mongodb_connection():
    """Test MongoDB connection and basic operations"""
    
    print("=" * 60)
    print("🔍 MONGODB CONNECTION TEST")
    print("=" * 60)
    
    # Step 1: Try to connect
    print(f"\n1️⃣ Connecting to MongoDB...")
    print(f"   URI: {MONGO_URI}")
    
    try:
        # Create client with timeout
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        
        # Test connection by getting server info
        client.server_info()
        print("   ✅ Connected successfully!")
        
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        print("\n   Possible issues:")
        print("   • MongoDB is not installed")
        print("   • MongoDB service is not running")
        print("   • Wrong connection URI")
        print("   • Firewall blocking the connection")
        return False
    
    # Step 2: List all databases
    print(f"\n2️⃣ Listing databases...")
    try:
        databases = client.list_database_names()
        print(f"   ✅ Databases found: {databases}")
    except Exception as e:
        print(f"   ❌ Failed to list databases: {e}")
    
    # Step 3: Access/Create database
    print(f"\n3️⃣ Accessing database: '{DATABASE_NAME}'")
    db = client[DATABASE_NAME]
    print(f"   ✅ Database accessed (will be created on first insert)")
    
    # Step 4: Access/Create collection
    print(f"\n4️⃣ Accessing collection: '{COLLECTION_NAME}'")
    collection = db[COLLECTION_NAME]
    print(f"   ✅ Collection accessed (will be created on first insert)")
    
    # Step 5: Insert test data
    print(f"\n5️⃣ Inserting test data...")
    test_document = {
        "test_id": "test_001",
        "message": "This is a test document",
        "timestamp": datetime.utcnow(),
        "tags": ["test", "mongodb", "connection"],
        "value": 123.45
    }
    
    try:
        result = collection.insert_one(test_document)
        print(f"   ✅ Document inserted with ID: {result.inserted_id}")
    except Exception as e:
        print(f"   ❌ Failed to insert: {e}")
        return False
    
    # Step 6: Retrieve test data
    print(f"\n6️⃣ Retrieving test data...")
    try:
        retrieved = collection.find_one({"test_id": "test_001"})
        if retrieved:
            print(f"   ✅ Document retrieved successfully!")
            print(f"   📄 Document: {retrieved}")
        else:
            print(f"   ❌ Document not found")
    except Exception as e:
        print(f"   ❌ Failed to retrieve: {e}")
    
    # Step 7: Count documents
    print(f"\n7️⃣ Counting documents...")
    try:
        count = collection.count_documents({})
        print(f"   ✅ Total documents in collection: {count}")
    except Exception as e:
        print(f"   ❌ Failed to count: {e}")
    
    # Step 8: Clean up - delete test data
    print(f"\n8️⃣ Cleaning up test data...")
    try:
        result = collection.delete_many({"test_id": "test_001"})
        print(f"   ✅ Deleted {result.deleted_count} test document(s)")
    except Exception as e:
        print(f"   ❌ Failed to delete: {e}")
    
    # Step 9: Close connection
    print(f"\n9️⃣ Closing connection...")
    client.close()
    print("   ✅ Connection closed")
    
    print("\n" + "=" * 60)
    print("✅ TEST COMPLETED")
    print("=" * 60)
    return True

def check_mongodb_service():
    """Check if MongoDB service is running (platform specific)"""
    import platform
    import subprocess
    
    system = platform.system()
    print(f"\n🖥️  System: {system}")
    
    if system == "Windows":
        try:
            result = subprocess.run(["sc", "query", "MongoDB"], capture_output=True, text=True)
            if "RUNNING" in result.stdout:
                print("✅ MongoDB service is RUNNING")
            elif "does not exist" in result.stdout:
                print("❌ MongoDB service not installed")
            else:
                print("⚠️  MongoDB service status unknown")
        except:
            print("⚠️  Could not check service status")
    
    elif system == "Linux":
        try:
            result = subprocess.run(["systemctl", "is-active", "mongod"], capture_output=True, text=True)
            if "active" in result.stdout:
                print("✅ MongoDB service is RUNNING")
            else:
                print("❌ MongoDB service is NOT running")
        except:
            print("⚠️  Could not check service status")
    
    elif system == "Darwin":  # macOS
        try:
            result = subprocess.run(["brew", "services", "list"], capture_output=True, text=True)
            if "mongodb-community" in result.stdout and "started" in result.stdout:
                print("✅ MongoDB service is RUNNING")
            else:
                print("❌ MongoDB service is NOT running")
        except:
            print("⚠️  Could not check service status")

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 MONGODB CONNECTION TESTER")
    print("=" * 60)
    
    # Check if MongoDB service is running
    check_mongodb_service()
    
    # Run the connection test
    success = test_mongodb_connection()
    
    if success:
        print("\n✅ Your MongoDB is working perfectly!")
        print("   You can now use it in your application.")
    else:
        print("\n❌ MongoDB connection failed.")
        print("\n📋 Troubleshooting steps:")
        print("   1. Install MongoDB from https://www.mongodb.com/try/download/community")
        print("   2. Start MongoDB service:")
        print("      • Windows: net start MongoDB")
        print("      • Mac: brew services start mongodb-community")
        print("      • Linux: sudo systemctl start mongod")
        print("   3. Check your .env file has correct MONGO_URI")
        print("   4. Make sure no firewall is blocking port 27017")