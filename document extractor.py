import streamlit as st
import pytesseract
from pdf2image import convert_from_bytes
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import os
import tempfile
import numpy as np
import re
import torch

# Chroma
os.environ["CHROMA_ALREADY_INITIALIZED"] = "false"
import chromadb
from chromadb.config import Settings

@st.cache_data(show_spinner=False)
def extract_text_from_pdf(pdf_file):
    images = convert_from_bytes(pdf_file.read())
    pages_text = []

    for idx, image in enumerate(images):
        text = pytesseract.image_to_string(image)
        pages_text.append((text, idx + 1))

    return pages_text

def improved_split_text(pages_text, chunk_size=500, overlap=100):
    all_chunks = []
    for page_text, page_num in pages_text:
        text = re.sub(r'\s+', ' ', page_text)
        text = re.sub(r'\n+', '\n', text)
        paragraphs = [p for p in text.split('\n') if p.strip()]
        chunks, current_chunk, current_size = [], [], 0

        for para in paragraphs:
            words = para.split()
            if current_size + len(words) <= chunk_size:
                current_chunk.extend(words)
                current_size += len(words)
            else:
                if current_chunk:
                    all_chunks.append({"text": " ".join(current_chunk), "page": page_num})

                current_chunk = current_chunk[-overlap:] if len(current_chunk) > overlap else []
                current_size = len(current_chunk)

                if len(words) <= chunk_size:
                    current_chunk.extend(words)
                    current_size += len(words)
                else:
                    for i in range(0, len(words), chunk_size - overlap):
                        chunk = words[i:i + chunk_size]
                        all_chunks.append({"text": " ".join(chunk), "page": page_num})
                    current_chunk = words[-overlap:] if len(words) > overlap else words
                    current_size = len(current_chunk)

        if current_chunk:
            all_chunks.append({"text": " ".join(current_chunk), "page": page_num})

    return all_chunks

def hybrid_search(query, chunks, embedding_model, collection, k=5):
    query_embedding = embedding_model.encode([query])[0].tolist()
    vector_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(k*2, len(chunks)),
        include=["documents", "metadatas", "distances"]
    )

    vector_chunks = vector_results["documents"][0]
    vector_metadatas = vector_results["metadatas"][0]

    keyword_matches = {}
    query_words = set(word.lower() for word in query.split() if len(word) > 3)

    for i, chunk in enumerate(vector_chunks):
        text = chunk.lower()
        score = sum(1 for word in query_words if word in text)
        keyword_matches[i] = score

    combined_results = []
    for i, (chunk, metadata) in enumerate(zip(vector_chunks, vector_metadatas)):
        combined_score = (k*2 - i) + keyword_matches.get(i, 0) * 2
        combined_results.append({
            "text": chunk,
            "page": metadata["page"],
            "score": combined_score
        })

    return sorted(combined_results, key=lambda x: x["score"], reverse=True)[:k]

def assemble_context(ranked_chunks):
    return "\n\n".join(f"[Page {chunk['page']}] {chunk['text']}" for chunk in ranked_chunks)

# === Main Streamlit App ===
st.set_page_config(page_title="PDF QA", layout="wide")
st.title("📄 SmartDoc QA")

model_choice = st.selectbox("Choose a language model:", ["FLAN-T5 Small", "Mistral (GPU only)"])

try:
    chroma_client = chromadb.EphemeralClient()
except:
    chroma_client = chromadb.Client()

uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file:
    file_name = uploaded_file.name
    if "current_file" not in st.session_state or st.session_state["current_file"] != file_name:
        for key in ["pages_text", "chunks", "collection", "embedding_model", "tokenizer", "flan_model"]:
            st.session_state.pop(key, None)
        st.session_state["current_file"] = file_name

    if "pages_text" not in st.session_state:
        with st.spinner("🔍 Extracting text from PDF..."):
            st.session_state["pages_text"] = extract_text_from_pdf(uploaded_file)
            st.success(f"✅ Text extracted from {len(st.session_state['pages_text'])} pages.")

    if "chunks" not in st.session_state:
        with st.spinner("⚙️ Creating chunks..."):
            st.session_state["chunks"] = improved_split_text(st.session_state["pages_text"])
            st.success(f"✅ Created {len(st.session_state['chunks'])} chunks!")

    if "embedding_model" not in st.session_state:
        with st.spinner("🧠 Loading embedding model..."):
            st.session_state["embedding_model"] = SentenceTransformer("paraphrase-MiniLM-L6-v2")
            st.success("✅ Embedding model loaded.")

    if "collection" not in st.session_state:
        with st.spinner("🗃️ Creating vector database..."):
            try:
                chroma_client.delete_collection(name="pdf_chunks")
            except:
                pass
            collection = chroma_client.create_collection(name="pdf_chunks", embedding_function=None)
            chunks = st.session_state["chunks"]
            chunk_texts = [chunk["text"] for chunk in chunks]
            metadatas = [{"page": chunk["page"]} for chunk in chunks]
            embeddings = st.session_state["embedding_model"].encode(chunk_texts).tolist()
            collection.add(
                documents=chunk_texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=[f"id_{i}" for i in range(len(chunks))]
            )
            st.session_state["collection"] = collection
            st.success("✅ Vector database created.")

    if "flan_model" not in st.session_state:
        with st.spinner("🔄 Loading language model..."):
            if model_choice == "FLAN-T5 Small":
                tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
                model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
            else:
                tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
                model = AutoModelForCausalLM.from_pretrained(
                    "mistralai/Mistral-7B-Instruct-v0.1", device_map="auto", torch_dtype=torch.float16
                )
            st.session_state["tokenizer"] = tokenizer
            st.session_state["flan_model"] = model
            st.session_state["is_causal"] = model_choice != "FLAN-T5 Small"
            st.success("✅ Language model loaded.")

    st.markdown("Ask a question about your PDF")
    query = st.text_input("Type your question here:")

    if query:
        with st.spinner("🤔 Finding answer..."):
            embedding_model = st.session_state["embedding_model"]
            collection = st.session_state["collection"]
            tokenizer = st.session_state["tokenizer"]
            model = st.session_state["flan_model"]
            is_causal = st.session_state["is_causal"]
            chunks = st.session_state["chunks"]

            expand_prompt = f"Expand this query with more terms: {query}"
            inputs = tokenizer(expand_prompt, return_tensors="pt", truncation=True, max_length=128)
            outputs = model.generate(**inputs, max_new_tokens=50)
            expanded_query = tokenizer.decode(outputs[0], skip_special_tokens=True)

            search_query = f"{query} {expanded_query}"
            top_chunks = hybrid_search(search_query, chunks, embedding_model, collection, k=5)
            context = assemble_context(top_chunks)
            cited_pages = [chunk["page"] for chunk in top_chunks]

            prompt = (
                f"Answer the question using only the context. If not found, say 'I cannot find information about this.'\n\n"
                f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
            )

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            outputs = model.generate(**inputs, max_new_tokens=200)
            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        st.markdown("### ✅ Answer")
        st.write(answer)
        st.markdown(f"*Sources: Pages {', '.join(map(str, set(cited_pages)))}*")
        with st.expander("View retrieved context"):
            st.text_area("Context used:", context, height=300)
