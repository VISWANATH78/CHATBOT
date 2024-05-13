import chromadb
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import PyPDF2

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as f:
        pdf_reader = PyPDF2.PdfReader(f)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to embed text
def embed_text(text):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.tolist()

# Initialize ChromaDB client
chroma_client = chromadb.Client()

# Create collection
collection = chroma_client.create_collection(name="data")

# Path to the PDF file
pdf_path = "jemh101.pdf"

# Extract text from PDF
pdf_text = extract_text_from_pdf(pdf_path)

# Initialize SentenceTransformer with LLaMA
sentence_transformer = SentenceTransformer('paraphrase-xlm-r-multilingual-v1')

# Encode text into embeddings
embedding = sentence_transformer.encode(pdf_text)

print(embedding)

# Add PDF content to collection
collection.add(
    documents=[pdf_text],
    ids=[pdf_path]  # You can use the PDF file path as the ID
)

# Query the collection to retrieve documents
results = collection.query(
    query_texts=[""],  # Query for all documents
    n_results=1  # Specify the number of results to retrieve
)

# Print the documents and their associated metadata
for result in results:
    document = result
    print("Document ID:", result)
    print("Document content:", results[result])
