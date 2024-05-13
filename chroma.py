from chromadb import Client

# Replace with your ChromaDB connection details
client = Client(host="localhost", port=8500)  # Update host and port if needed

collection = client.get_collection(name="data")

# Get all documents in the collection (might be large for many documents)
documents = collection.get_all()

# Print some information from each document
for doc in documents:
    print(f"Document ID: {doc.id}")
    # Access document content based on its format (might require additional processing)
    print(f"Document content (example): {doc.content[:100]}...")  # Print first 100 characters
