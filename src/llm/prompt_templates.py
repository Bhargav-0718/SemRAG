from typing import List, Dict, Any


class PromptTemplates:
    """Collection of prompt templates for various SemRAG tasks."""
    
    # Entity Extraction Prompts
    ENTITY_EXTRACTION = """Extract entities and their relationships from the following text chunk.

Text:
{text}

Extract the following types of entities:
- PERSON: Names of people
- ORGANIZATION: Companies, institutions, groups
- LOCATION: Places, geographical locations
- EVENT: Historical or significant events
- CONCEPT: Abstract ideas, theories, philosophies
- DATE: Time references

For each entity, also identify relationships with other entities mentioned in the text.

Provide your response in the following JSON format:
{{
    "entities": [
        {{
            "name": "entity name",
            "type": "ENTITY_TYPE",
            "description": "brief description"
        }}
    ],
    "relationships": [
        {{
            "source": "entity1",
            "target": "entity2",
            "relationship": "relationship type",
            "description": "brief description"
        }}
    ]
}}
"""

    # Chunk Summarization Prompt
    CHUNK_SUMMARY = """Summarize the following text chunk in a concise manner, capturing the key information and main ideas.

Text:
{text}

Provide a summary in 2-3 sentences that captures the essence of this content.
"""

    # Community Summarization Prompt
    COMMUNITY_SUMMARY = """You are analyzing a community of related content chunks from a document about Dr. B.R. Ambedkar.

The community contains the following elements:

Entities: {entities}

Chunk Summaries:
{chunk_summaries}

Provide a comprehensive summary (3-5 sentences) that:
1. Identifies the main theme connecting these chunks
2. Highlights key entities and their relationships
3. Captures the most important information from this community

Summary:
"""

    # Local Search (Entity-based) Prompt
    LOCAL_SEARCH_PROMPT = """You are answering a question about Dr. B.R. Ambedkar using specific contextual information.

Question: {question}

Relevant Context:
{context}

Key Entities: {entities}

Using the provided context and entity information, provide a detailed and accurate answer to the question.
Ensure your answer is:
1. Directly based on the provided context
2. Accurate and factual
3. Well-structured and clear
4. Includes specific details from the context

If the context doesn't contain enough information to fully answer the question, acknowledge this limitation.

Answer:
"""

    # Global Search (Community-based) Prompt
    GLOBAL_SEARCH_PROMPT = """You are answering a high-level question about Dr. B.R. Ambedkar using community summaries.

Question: {question}

Community Summaries:
{community_summaries}

Based on these thematic summaries, provide a comprehensive answer that:
1. Synthesizes information across multiple communities
2. Provides a broad perspective on the topic
3. Is well-organized and coherent
4. Addresses the question thoroughly

Answer:
"""

    # Hybrid Search Prompt
    HYBRID_SEARCH_PROMPT = """You are answering a question about Dr. B.R. Ambedkar using both detailed context and high-level summaries.

Question: {question}

=== Detailed Context (from specific chunks) ===
{local_context}

=== High-Level Summaries (from communities) ===
{global_context}

=== Key Entities ===
{entities}

Provide a comprehensive answer that:
1. Combines specific details from the context with broader themes from summaries
2. Is well-structured and flows naturally
3. Balances specificity with overall understanding
4. Cites specific information when relevant

Answer:
"""

    # Relationship Extraction Prompt
    RELATIONSHIP_EXTRACTION = """Analyze the following text and identify relationships between entities.

Text: {text}

Identified Entities: {entities}

For each pair of entities that have a meaningful relationship in this text, describe:
1. The type of relationship (e.g., "worked_with", "founded", "influenced", "opposed")
2. A brief description of the relationship
3. The strength/importance of this relationship (high/medium/low)

Provide your response as a JSON array of relationships.
"""

    @staticmethod
    def format_entity_extraction(text: str) -> str:
        """Format entity extraction prompt."""
        return PromptTemplates.ENTITY_EXTRACTION.format(text=text)
    
    @staticmethod
    def format_chunk_summary(text: str) -> str:
        """Format chunk summarization prompt."""
        return PromptTemplates.CHUNK_SUMMARY.format(text=text)
    
    @staticmethod
    def format_community_summary(entities: List[str], chunk_summaries: List[str]) -> str:
        """Format community summarization prompt."""
        entities_str = ", ".join(entities)
        summaries_str = "\n".join([f"- {s}" for s in chunk_summaries])
        return PromptTemplates.COMMUNITY_SUMMARY.format(
            entities=entities_str,
            chunk_summaries=summaries_str
        )
    
    @staticmethod
    def format_local_search(question: str, context: List[str], entities: List[str]) -> str:
        """Format local search prompt."""
        context_str = "\n\n".join([f"[Context {i+1}]\n{c}" for i, c in enumerate(context)])
        entities_str = ", ".join(entities)
        return PromptTemplates.LOCAL_SEARCH_PROMPT.format(
            question=question,
            context=context_str,
            entities=entities_str
        )
    
    @staticmethod
    def format_global_search(question: str, community_summaries: List[str]) -> str:
        """Format global search prompt."""
        summaries_str = "\n\n".join([f"[Community {i+1}]\n{s}" for i, s in enumerate(community_summaries)])
        return PromptTemplates.GLOBAL_SEARCH_PROMPT.format(
            question=question,
            community_summaries=summaries_str
        )
    
    @staticmethod
    def format_hybrid_search(
        question: str, 
        local_context: List[str], 
        global_context: List[str],
        entities: List[str]
    ) -> str:
        """Format hybrid search prompt."""
        local_str = "\n\n".join([f"[Chunk {i+1}]\n{c}" for i, c in enumerate(local_context)])
        global_str = "\n\n".join([f"[Community {i+1}]\n{s}" for i, s in enumerate(global_context)])
        entities_str = ", ".join(entities)
        return PromptTemplates.HYBRID_SEARCH_PROMPT.format(
            question=question,
            local_context=local_str,
            global_context=global_str,
            entities=entities_str
        )
