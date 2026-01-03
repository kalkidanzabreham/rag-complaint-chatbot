import pandas as pd
from sklearn.model_selection import train_test_split
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from tqdm import tqdm

df = pd.read_csv("data/processed/filtered_complaints.csv")

# Stratified Sampling (10k–15k)
sample_size = 12000

df_sample, _ = train_test_split(
    df,
    train_size=sample_size,
    stratify=df['Product'],
    random_state=42
)

print(df_sample['Product'].value_counts())

# Chunking Strategy

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# Load Embedding Model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize ChromaDB
client = chromadb.Client(
    chromadb.config.Settings(
        persist_directory="vector_store",
        anonymized_telemetry=False
    )
)

collection = client.get_or_create_collection(
    name="complaints"
)

# Chunk → Embed → Store
for idx, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
    chunks = text_splitter.split_text(row['clean_narrative'])

    embeddings = embedding_model.encode(chunks)

    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        collection.add(
            documents=[chunk],
            embeddings=[emb.tolist()],
            metadatas=[{
                "complaint_id": row.get("Complaint ID", idx),
                "product": row["Product"],
                "issue": row.get("Issue"),
                "company": row.get("Company"),
                "chunk_index": i
            }],
            ids=[f"{idx}_{i}"]
        )

print("Vector store created and persisted successfully!")
print("Vector store created successfully!")
