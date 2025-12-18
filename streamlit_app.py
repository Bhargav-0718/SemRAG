"""
Streamlit UI for AmbedkarGPT - SemRAG System
Interactive interface for querying the Ambedkar knowledge base
"""

import streamlit as st
import os
from pathlib import Path
from src.pipeline.ambedkargpt import AmbedkarGPT
import logging

# Configure page
st.set_page_config(
    page_title="AmbedkarGPT - SemRAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-title {
        color: #1f4788;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 0.5em;
    }
    .subtitle {
        color: #666;
        font-size: 1.1em;
        margin-bottom: 1em;
    }
    .search-type-btn {
        margin: 0.25em;
    }
    .response-box {
        background-color: #f0f2f6;
        padding: 1.5em;
        border-radius: 0.5em;
        margin-top: 1em;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "rag_system" not in st.session_state:
    st.session_state.rag_system = None
    st.session_state.data_loaded = False
    st.session_state.processing = False

# Sidebar configuration
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    # Document processing section
    st.markdown("### 📄 Document Processing")
    
    if st.button("🔄 Load Processed Data", help="Load pre-processed data and embeddings from cache"):
        with st.spinner("Loading processed data and embeddings..."):
            try:
                rag_system = AmbedkarGPT(config_path="config.yaml")
                rag_system.load_processed_data()
                st.session_state.rag_system = rag_system
                st.session_state.data_loaded = True
                st.success("✅ Data loaded successfully!")
                st.info("📊 Embeddings loaded from cache (instant access)")
            except Exception as e:
                st.error(f"❌ Error loading data: {e}")
    
    if st.button("🔨 Process Document", help="Process a new PDF document through the entire pipeline"):
        pdf_path = "data/Ambedkar_book.pdf"
        if Path(pdf_path).exists():
            with st.spinner("Processing document (this may take several minutes on first run)..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    rag_system = AmbedkarGPT(config_path="config.yaml")
                    status_text.text("📄 Processing PDF and extracting text...")
                    progress_bar.progress(10)
                    
                    rag_system.process_document(pdf_path=pdf_path)
                    status_text.text("✅ Document processing complete!")
                    progress_bar.progress(100)
                    
                    st.session_state.rag_system = rag_system
                    st.session_state.data_loaded = True
                    st.success("✅ Document processed and embeddings cached!")
                    st.info("💾 Data saved to data/processed/ | 🚀 Next runs will be instant with cached embeddings")
                except Exception as e:
                    st.error(f"❌ Error processing document: {e}")
        else:
            st.error(f"❌ PDF not found: {pdf_path}")
    
    st.divider()
    
    # Search configuration
    st.markdown("### 🔍 Search Configuration")
    search_type = st.radio(
        "Search Type:",
        ["Local", "Global", "Hybrid"],
        help="""
        - **Local**: Entity-based, specific details
        - **Global**: Community-based, thematic overview
        - **Hybrid**: Best of both (recommended)
        """
    )
    
    st.divider()
    
    # System info
    st.markdown("### ℹ️ System Info")
    if st.session_state.data_loaded and st.session_state.rag_system:
        rag = st.session_state.rag_system
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Chunks", len(rag.chunks) if rag.chunks else 0)
            st.metric("Entities", len(rag.entities) if rag.entities else 0)
        with col2:
            st.metric("Communities", len(rag.communities) if rag.communities else 0)
            if rag.chunk_summaries:
                st.metric("Summaries", len(rag.chunk_summaries))
    else:
        st.warning("⚠️ No data loaded. Click 'Load Processed Data' or 'Process Document' to get started.")
    
    st.divider()
    st.markdown("""
    ### 📖 About
    **AmbedkarGPT** uses SemRAG (Semantic Knowledge-Augmented RAG) to provide intelligent 
    Q&A on Dr. B.R. Ambedkar's works.
    
    - 🔗 [SemRAG Paper](https://arxiv.org/abs/2507.21110)
    - 🌐 [GitHub](https://github.com/Bhargav-0718/SemRAG)
    """)

# Main content
st.markdown('<h1 class="main-title">📚 AmbedkarGPT</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Semantic Knowledge-Augmented RAG for Ambedkar Studies</p>', unsafe_allow_html=True)

if not st.session_state.data_loaded:
    st.warning("⚠️ Please load or process data using the sidebar configuration panel first.")
else:
    # Query interface
    st.markdown("## 🤔 Ask a Question")
    
    query = st.text_area(
        "Enter your question about Dr. Ambedkar and his works:",
        placeholder="e.g., What were Dr. Ambedkar's views on social justice?",
        height=100
    )
    
    col1, col2, col3 = st.columns(3)
    with col1:
        submit_btn = st.button("🚀 Search", use_container_width=True)
    with col2:
        clear_btn = st.button("🔄 Clear", use_container_width=True)
    with col3:
        st.empty()
    
    if clear_btn:
        st.rerun()
    
    if submit_btn and query.strip():
        with st.spinner(f"Searching using {search_type} search..."):
            try:
                # Map UI search type to system search type
                search_type_map = {
                    "Local": "local",
                    "Global": "global",
                    "Hybrid": "hybrid"
                }
                
                result = st.session_state.rag_system.query(
                    question=query,
                    search_type=search_type_map[search_type]
                )
                
                # Display results
                st.markdown("## 📊 Results")
                
                # Answer
                st.markdown("### 💡 Answer")
                st.markdown(f'<div class="response-box">{result["answer"]}</div>', unsafe_allow_html=True)
                
                # Tabs for detailed results
                tab1, tab2, tab3 = st.tabs(["📄 Sources", "🏷️ Entities", "📈 Details"])
                
                with tab1:
                    st.markdown("### Source Chunks")
                    if "retrieved_chunks" in result and result["retrieved_chunks"]:
                        for i, chunk in enumerate(result["retrieved_chunks"][:5], 1):
                            with st.expander(f"Source {i}"):
                                st.write(chunk)
                    else:
                        st.info("No specific chunks retrieved")
                
                with tab2:
                    st.markdown("### Relevant Entities")
                    if "entities" in result and result["entities"]:
                        entity_col1, entity_col2 = st.columns(2)
                        for i, entity in enumerate(result["entities"]):
                            if i % 2 == 0:
                                entity_col1.metric(f"Entity {i+1}", entity)
                            else:
                                entity_col2.metric(f"Entity {i+1}", entity)
                    else:
                        st.info("No entities found")
                
                with tab3:
                    st.markdown("### Search Metadata")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if "num_candidates" in result:
                            st.metric("Candidate Chunks", result["num_candidates"])
                    
                    with col2:
                        st.metric("Search Type", search_type.upper())
                    
                    with col3:
                        if "search_stats" in result:
                            st.metric("Relevant Items", len(result.get("entities", [])))
                
                st.divider()
                st.success("✅ Query processing complete!")
                
            except Exception as e:
                st.error(f"❌ Error processing query: {e}")
                st.info("💡 Tip: Make sure you've loaded or processed data first.")
    elif submit_btn:
        st.warning("⚠️ Please enter a question.")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9em;">
    <p>Powered by SemRAG | Built with Streamlit | OpenAI Embeddings & GPT</p>
    <p>© 2025 AmbedkarGPT | Data: Dr. B.R. Ambedkar's Works</p>
</div>
""", unsafe_allow_html=True)
