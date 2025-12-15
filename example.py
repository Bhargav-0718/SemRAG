"""Example usage of AmbedkarGPT system."""

import json
import os
from src.pipeline.ambedkargpt import AmbedkarGPT

def main():
    """Run example queries on the AmbedkarGPT system."""
    
    # Initialize the system
    print("Initializing AmbedkarGPT...")
    rag_system = AmbedkarGPT(config_path="config.yaml")
    
    # Check if processed data exists and is usable
    chunks_path = "data/processed/chunks.json"
    processed_data_exists = os.path.exists(chunks_path) and os.path.getsize(chunks_path) > 0
    
    if not processed_data_exists:
        print("\nProcessing document (this may take several minutes)...")
        print("Steps:")
        print("1. Loading PDF")
        print("2. Semantic chunking")
        print("3. Extracting entities")
        print("4. Building knowledge graph")
        print("5. Detecting communities")
        print("6. Generating summaries")
        print("7. Initializing retrieval")
        
        # Process the Ambedkar book
        rag_system.process_document(pdf_path="data/Ambedkar_book.pdf")
        
        print("\n✓ Document processed successfully!")
        print("Processed data saved to data/processed/")
    else:
        print("\nLoading previously processed data...")
        try:
            rag_system.load_processed_data()
            print("✓ Data loaded successfully!")
        except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
            print(f"⚠️  Processed data invalid or missing ({exc}); reprocessing document...\n")
            rag_system.process_document(pdf_path="data/Ambedkar_book.pdf")
            print("\n✓ Document reprocessed successfully!")
            print("Processed data saved to data/processed/")
    
    # Example questions demonstrating different search types
    questions = [
        {
            "question": "What were Dr. Ambedkar's views on social justice?",
            "search_type": "local",
            "description": "LOCAL SEARCH (Entity-based) - Best for specific details"
        },
        {
            "question": "What were the main themes in Ambedkar's philosophy?",
            "search_type": "global",
            "description": "GLOBAL SEARCH (Community-based) - Best for broad themes"
        },
        {
            "question": "How did Ambedkar's personal experiences influence his political work?",
            "search_type": "hybrid",
            "description": "HYBRID SEARCH (Combined) - Best for complex questions"
        }
    ]
    
    # Query the system
    print("\n" + "="*80)
    print("EXAMPLE QUERIES")
    print("="*80)
    
    for i, q in enumerate(questions, 1):
        print(f"\n{'─'*80}")
        print(f"Question {i}: {q['question']}")
        print(f"Search Type: {q['description']}")
        print("─"*80)
        
        # Query the system
        result = rag_system.query(
            question=q["question"],
            search_type=q["search_type"]
        )
        
        # Display results
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nSources used: {len(result.get('context', []))} chunks")
        print(f"Entities referenced: {', '.join(result.get('entities', [])[:5])}")
        if len(result.get('entities', [])) > 5:
            print(f"  ... and {len(result['entities']) - 5} more")
    
    print("\n" + "="*80)
    print("Interactive Mode")
    print("="*80)
    print("You can now ask your own questions!")
    print("Type 'quit' or 'exit' to end the session.\n")
    
    while True:
        # Get user input
        user_question = input("Your question: ").strip()
        
        if user_question.lower() in ['quit', 'exit', 'q']:
            print("Thank you for using AmbedkarGPT!")
            break
        
        if not user_question:
            continue
        
        # Ask for search type
        print("Search type: [1] Local (specific), [2] Global (broad), [3] Hybrid (default)")
        search_choice = input("Choose (1/2/3 or press Enter for hybrid): ").strip()
        
        search_type_map = {
            '1': 'local',
            '2': 'global',
            '3': 'hybrid',
            '': 'hybrid'
        }
        search_type = search_type_map.get(search_choice, 'hybrid')
        
        # Query the system
        print(f"\nSearching ({search_type})...")
        result = rag_system.query(
            question=user_question,
            search_type=search_type
        )
        
        # Display result
        print(f"\n{'-'*80}")
        print(f"Answer:\n{result['answer']}")
        print(f"\nSources: {len(result.get('context', []))} chunks")
        if result.get('entities'):
            print(f"Key entities: {', '.join(result['entities'][:5])}")
        print('-'*80 + "\n")

if __name__ == "__main__":
    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not found in environment variables.")
        print("Please set it before running:")
        print("  export OPENAI_API_KEY='your-api-key-here'")
        print("\nOr create a .env file with:")
        print("  OPENAI_API_KEY=your-api-key-here")
        exit(1)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
