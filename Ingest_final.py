from chromadb import Client, Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import logging
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from langchain.docstore.document import Document
from PyPDF2 import PdfReader

ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
SOURCE_DIRECTORY = f"{ROOT_DIRECTORY}/SOURCE_DOCUMENTS"
INGEST_THREADS = 4  # Adjust as needed
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL_NAME = "hkunlp/instructor-large"  # Model name
PERSIST_DIRECTORY = f"{ROOT_DIRECTORY}/chroma"
SAVE_MODEL_DIRECTORY = "/nlsasfs/home/mcq/kvijay/local3/localGPT/models/embedding"

CHROMA_SETTINGS = Settings(
    anonymized_telemetry=False,
    is_persistent=True,
    persist_directory=PERSIST_DIRECTORY
)

# Define PDF loader class
class PdfLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        try:
            with open(self.file_path, "rb") as f:
                pdf_reader = PdfReader(f)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
            file_name = os.path.basename(self.file_path)
            return [Document(text=text, page_content=text, metadata={"source": file_name})]
        except Exception as ex:
            logging.error(f"Error loading PDF file {self.file_path}: {ex}")
            return []

# Define DOCUMENT_MAP
DOCUMENT_MAP = {".pdf": PdfLoader}

def load_single_document(file_path: str) -> Document:
    try:
        file_extension = os.path.splitext(file_path)[1]
        loader_class = DOCUMENT_MAP.get(file_extension)
        if loader_class:
            logging.info(f"{file_path} loaded.")
            loader = loader_class(file_path)
        else:
            logging.error(f"{file_path} document type is undefined.")
            raise ValueError("Document type is undefined")
        return loader.load()[0]
    except Exception as ex:
        logging.error(f"{file_path} loading error: {ex}")
        return None

def load_document_batch(filepaths):
    logging.info("Loading document batch")
    with ThreadPoolExecutor(len(filepaths)) as exe:
        futures = [exe.submit(load_single_document, name) for name in filepaths]
        if futures is None:
            logging.error("Failed to submit tasks.")
            return None
        else:
            data_list = [future.result() for future in futures]
            return (data_list, filepaths)

def load_documents(source_dir: str) -> list[Document]:
    paths = []
    for root, _, files in os.walk(source_dir):
        for file_name in files:
            logging.info(f"Importing: {file_name}")
            file_extension = os.path.splitext(file_name)[1]
            source_file_path = os.path.join(root, file_name)
            if file_extension in DOCUMENT_MAP.keys():
                paths.append(source_file_path)

    n_workers = min(INGEST_THREADS, max(len(paths), 1))
    chunksize = round(len(paths) / n_workers)
    docs = []
    with ProcessPoolExecutor(n_workers) as executor:
        futures = []
        for i in range(0, len(paths), chunksize):
            filepaths = paths[i: (i + chunksize)]
            try:
                future = executor.submit(load_document_batch, filepaths)
            except Exception as ex:
                logging.error(f"Executor task failed: {ex}")
                future = None
            if future is not None:
                futures.append(future)
        for future in as_completed(futures):
            try:
                contents, _ = future.result()
                docs.extend(contents)
            except Exception as ex:
                logging.error(f"Exception: {ex}")
    return docs

def split_documents(documents: list[Document]) -> list[Document]:
    text_docs = []
    for doc in documents:
        if doc is not None:
            text_docs.append(doc)
    return text_docs

def load_or_download_model():
    # Check if the model has been saved locally
    if os.path.exists(SAVE_MODEL_DIRECTORY):
        # Load the model from the specified directory
        model = SentenceTransformer(SAVE_MODEL_DIRECTORY)
    else:
        try:
            # Initialize the model
            model = SentenceTransformer(EMBEDDING_MODEL_NAME)

            # Save the model to the specified directory
            model.save(SAVE_MODEL_DIRECTORY)
        except Exception as ex:
            logging.error(f"Error loading and saving SentenceTransformer model: {ex}")
            return None
    return model

def main():
    logging.info("Loading documents from SOURCE_DIRECTORY")
    documents = load_documents(SOURCE_DIRECTORY)
    text_documents = split_documents(documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    texts = text_splitter.split_documents(text_documents)
    logging.info(f"Loaded {len(documents)} documents from SOURCE_DIRECTORY")
    logging.info(f"Split into {len(texts)} chunks of text")

    # Debug: Print IDs and sources of text chunks
    for i, text_chunk in enumerate(texts):
        logging.info(f"Page {i + 1} - ID: {text_chunk.metadata['source']}, Source: {text_chunk.metadata['source']}")

    all_text = " ".join(text_chunk.page_content for text_chunk in texts)

    # Load or download the SentenceTransformer model
    model = load_or_download_model()
    if model is None:
        logging.error("Error loading SentenceTransformer model")
        return

    embeddings = model.encode(all_text)

    logging.info(f"Loaded embeddings from {EMBEDDING_MODEL_NAME}")

    logging.info(f"Number of embeddings: {len(embeddings)}")
    logging.info(f"Number of IDs: {len(texts)}")
    logging.info(f"First few embeddings: {embeddings[:5]}")
    logging.info(f"First few IDs: {[text_chunk.metadata['source'] for text_chunk in texts][:5]}")

    # Create a ChromaDB client
    chroma_client = Client(settings=CHROMA_SETTINGS)

    try:
        # Try to get the collection named "chromadb"
        collection = chroma_client.get_collection(name="data")
        logging.info("Collection chromadb exists")
    except ValueError:
        # If the collection does not exist, create it
        chroma_client.create_collection(name="data")
        logging.info("Created collection {name}")
        # Get the collection again
        collection = chroma_client.get_collection(name="data")

    # Add documents and embeddings to the collection
    for text_chunk, embedding in zip(texts, embeddings):
        doc_id = text_chunk.metadata["source"]
        source_data = text_chunk.page_content  # Assuming page_content contains the source data
        embedding_list = embedding.tolist()  # Convert NumPy array to list of floats

        # Add source data to the metadata
        metadata = {"source": doc_id, "source_data": source_data}

        # Debug: Print the document ID and embedding
        logging.info(f"Document ID: {doc_id}")
        logging.info(f"Embedding: {embedding_list}")

        # Add the document with metadata and embedding to the collection
        collection.add(ids=[doc_id], embeddings=[embedding_list], metadatas=[metadata])
        print("viswa-", doc_id, embedding_list)

    logging.info("Documents added to {name} collection")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    main()
