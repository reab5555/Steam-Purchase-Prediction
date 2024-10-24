from pymongo import MongoClient
import json
import sys
from datetime import datetime
import logging
from pprint import pprint

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_mongodb_connection(connection_string):
    """
    Test MongoDB connection before proceeding with data upload.

    Args:
        connection_string (str): MongoDB connection string
    Returns:
        bool: True if connection successful, False otherwise
    """
    try:
        # Use longer timeout for initial connection test
        client = MongoClient(
            connection_string,
            serverSelectionTimeoutMS=30000,  # 30 seconds for testing
            connectTimeoutMS=30000,
            socketTimeoutMS=30000,
            directConnection=False
        )

        # Test the connection
        logger.info("Attempting to connect to MongoDB...")
        client.admin.command('ping')

        # Get cluster information
        logger.info("Connection successful! Checking cluster information...")
        server_info = client.server_info()
        logger.info(f"Connected to MongoDB version: {server_info.get('version', 'unknown')}")

        # List available databases
        db_list = client.list_database_names()
        logger.info(f"Available databases: {db_list}")

        client.close()
        return True

    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        logger.info("Please verify:")
        logger.info("1. Your IP address is whitelisted in MongoDB Atlas")
        logger.info("2. Username and password are correct")
        logger.info("3. Network connectivity to MongoDB Atlas")
        return False


def preview_json_data(json_file_path, sample_size=5):
    """Previous preview_json_data function remains the same"""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        logger.info(f"Successfully loaded JSON file")
        total_documents = len(data) if isinstance(data, list) else 1
        logger.info(f"Total documents in JSON: {total_documents}")

        if isinstance(data, list):
            logger.info(f"\nPreviewing first {sample_size} documents:")
            for idx, doc in enumerate(data[:sample_size]):
                logger.info(f"\nDocument {idx + 1}:")
                pprint(doc)
        else:
            logger.info("Single document JSON:")
            pprint(data)

        return data
    except Exception as e:
        logger.error(f"Error previewing JSON file: {e}")
        raise


def load_json_to_mongodb(connection_string, database_name, collection_name, json_file_path):
    """Previous load_json_to_mongodb function remains the same but uses updated connection parameters"""
    try:
        # First test the connection
        logger.info("Testing MongoDB connection...")
        if not test_mongodb_connection(connection_string):
            raise Exception("Failed to establish MongoDB connection")

        # Preview the JSON data
        logger.info("Previewing JSON data...")
        data = preview_json_data(json_file_path)

        # Connect with full configuration
        client = MongoClient(
            connection_string,
            serverSelectionTimeoutMS=300000,
            connectTimeoutMS=300000,
            socketTimeoutMS=300000,
            maxPoolSize=None,
            waitQueueTimeoutMS=300000,
            retryWrites=True,
            w='majority',
            wTimeoutMS=300000,
            directConnection=False
        )

        db = client[database_name]
        collection = db[collection_name]

        # Rest of the function remains the same...
        # Add metadata and insert data as before
        if isinstance(data, list):
            for doc in data:
                doc['uploaded_at'] = datetime.utcnow()
        else:
            data['uploaded_at'] = datetime.utcnow()

        if isinstance(data, list):
            result = collection.insert_many(
                data,
                ordered=False,
                bypass_document_validation=True
            )
            logger.info(f"Successfully inserted {len(result.inserted_ids)} documents")
        else:
            result = collection.insert_one(
                data,
                bypass_document_validation=True
            )
            logger.info(f"Successfully inserted document with ID: {result.inserted_id}")

        logger.info("Data loading completed successfully")

    except Exception as e:
        logger.error(f"Error loading data to MongoDB: {e}")
        raise
    finally:
        if 'client' in locals():
            client.close()
            logger.info("MongoDB connection closed")


if __name__ == "__main__":
    # Updated connection string format for MongoDB Atlas
    CONNECTION_STRING = "mongodb+srv://username:password@..."
    DATABASE_NAME = "steam_reviews_db_test"
    COLLECTION_NAME = "steam_reviews_collection_test"
    JSON_FILE_PATH = R"/Steam Recommender/steam_reviews.json"

    try:
        load_json_to_mongodb(
            CONNECTION_STRING,
            DATABASE_NAME,
            COLLECTION_NAME,
            JSON_FILE_PATH
        )
    except Exception as e:
        logger.error(f"Script execution failed: {e}")
        sys.exit(1)