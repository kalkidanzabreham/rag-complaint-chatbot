# Intelligent Complaint Analysis for Financial Services  
## RAG-Powered Chatbot for Actionable Customer Insights

### Project Overview
CrediTrust Financial is a fast-growing digital finance company operating across East African markets. With thousands of customer complaints submitted monthly across multiple financial products, internal teams struggle to extract timely and actionable insights from unstructured complaint narratives.

This project builds an **Intelligent Complaint Analysis System** using **Retrieval-Augmented Generation (RAG)** to transform raw customer complaints into evidence-backed, searchable insights for internal stakeholders such as Product Managers, Customer Support, and Compliance teams.

The system enables users to ask natural language questions (e.g., *“Why are customers unhappy with credit cards?”*) and receive concise, grounded answers based on real complaint data.

---

## Business Objectives
The success of this project is measured by its ability to:

- Reduce the time required to identify emerging complaint trends from days to minutes  
- Enable non-technical teams to explore customer feedback without data analysts  
- Shift the organization from reactive issue handling to proactive, data-driven decision-making  

---

## Dataset
The project uses customer complaint data from the **Consumer Financial Protection Bureau (CFPB)**, which contains:

- Product and company metadata  
- Issue and sub-issue categories  
- Free-text consumer complaint narratives  
- Submission dates and geographic information  

### Target Product Categories
- Credit card  
- Personal loans  
- Checking or savings account  
- Money transfer, virtual currency, or money service  

---

## Project Structure
```bash
rag-complaint-chatbot/
│
├── data/
│ ├── raw/ # Original CFPB dataset
│ └── processed/ # Cleaned and filtered dataset
│ └── filtered_complaints.csv
│
├── notebooks/
│ └── task1_eda_preprocessing.ipynb
│
├── src/
│ └── task2_chunk_embed_index.py
│
├── vector_store/ # Persisted ChromaDB vector store
│
├── requirements.txt
├── README.md
└── .gitignore
```


---

## Task 1: Exploratory Data Analysis & Preprocessing
**Objective:** Understand the structure, quality, and characteristics of the complaint data and prepare it for semantic search.

### Key Steps
- Loaded the full CFPB complaint dataset  
- Analyzed complaint distribution across products  
- Examined narrative length distribution and missing values  
- Filtered complaints to target financial products  
- Removed records without complaint narratives  
- Cleaned text by:
  - Lowercasing  
  - Removing boilerplate phrases  
  - Removing special characters and extra whitespace  

### Output
- Cleaned and filtered dataset saved as:


data/processed/filtered_complaints.csv


---

## Task 2: Text Chunking, Embedding & Vector Store Indexing
**Objective:** Convert complaint narratives into vector embeddings suitable for semantic retrieval.

### Sampling Strategy
- Created a **stratified sample of 12,000 complaints**
- Ensured proportional representation across all target product categories

### Chunking Strategy
- Used `RecursiveCharacterTextSplitter`
- Chunk size: **500 characters**
- Chunk overlap: **50 characters**
- This balances semantic completeness with embedding efficiency

### Embedding Model
- **Model:** `sentence-transformers/all-MiniLM-L6-v2`
- Chosen for:
  - Strong semantic performance
  - Lightweight architecture
  - Industry-standard use in RAG systems

### Vector Store
- **Database:** ChromaDB
- Stored:
  - Text chunks
  - Vector embeddings
  - Metadata (product, issue, complaint ID, chunk index)
- Vector store persisted locally in:


vector_store/


---

## How to Run the Project

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```
2. Run Task 1 (EDA & Preprocessing)

Open and execute:
```bash
notebooks/task1_eda_preprocessing.ipynb

```
3. Run Task 2 (Chunking & Indexing)
```bash
python src/task2_chunk_embed_index.py
```

### Tools & Technologies
- Python
- Pandas, NumPy, Matplotlib, Seaborn
- Sentence Transformers
- LangChain
- ChromaDB

