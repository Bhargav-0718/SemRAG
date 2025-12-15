"""Entity extraction module using LLM.

Extracts entities and relationships from text chunks.
"""

import json
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class EntityExtractor:
    """Extract entities and relationships from text using LLM."""
    
    def __init__(self, llm_client, entity_types: List[str] = None):
        """Initialize entity extractor.
        
        Args:
            llm_client: LLM client for entity extraction
            entity_types: List of entity types to extract
        """
        self.llm_client = llm_client
        self.entity_types = entity_types or [
            "PERSON", "ORGANIZATION", "LOCATION", 
            "EVENT", "CONCEPT", "DATE"
        ]
    
    def extract_from_chunk(
        self, 
        chunk_text: str, 
        chunk_id: int
    ) -> Dict[str, Any]:
        """Extract entities and relationships from a single chunk.
        
        Args:
            chunk_text: Text content of the chunk
            chunk_id: Identifier for the chunk
            
        Returns:
            Dictionary with entities and relationships
        """
        from ..llm.prompt_templates import PromptTemplates
        
        prompt = PromptTemplates.format_entity_extraction(chunk_text)
        
        try:
            response = self.llm_client.generate_json(prompt, temperature=0.3)
            
            entities = response.get("entities", [])
            relationships = response.get("relationships", [])
            
            # Add chunk reference to each entity
            for entity in entities:
                entity["source_chunk"] = chunk_id
            
            # Add chunk reference to each relationship
            for rel in relationships:
                rel["source_chunk"] = chunk_id
            
            logger.info(f"Extracted {len(entities)} entities and {len(relationships)} relationships from chunk {chunk_id}")
            
            return {
                "chunk_id": chunk_id,
                "entities": entities,
                "relationships": relationships
            }
            
        except Exception as e:
            logger.error(f"Error extracting entities from chunk {chunk_id}: {e}")
            return {
                "chunk_id": chunk_id,
                "entities": [],
                "relationships": []
            }
    
    def extract_from_chunks(
        self, 
        chunks: List[Dict[str, Any]]
    ) -> Tuple[List[Dict], List[Dict]]:
        """Extract entities from multiple chunks.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Tuple of (all_entities, all_relationships)
        """
        logger.info(f"Extracting entities from {len(chunks)} chunks")
        
        all_entities = []
        all_relationships = []
        
        for chunk in chunks:
            chunk_id = chunk["chunk_id"]
            chunk_text = chunk.get("text_with_buffer", chunk.get("text", ""))
            
            result = self.extract_from_chunk(chunk_text, chunk_id)
            
            all_entities.extend(result["entities"])
            all_relationships.extend(result["relationships"])
        
        # Deduplicate entities by name (keep first occurrence)
        unique_entities = self._deduplicate_entities(all_entities)
        
        logger.info(f"Extracted total: {len(unique_entities)} unique entities, {len(all_relationships)} relationships")
        
        return unique_entities, all_relationships
    
    def _deduplicate_entities(self, entities: List[Dict]) -> List[Dict]:
        """Remove duplicate entities, keeping track of all source chunks.
        
        Args:
            entities: List of entity dictionaries
            
        Returns:
            List of deduplicated entities
        """
        entity_map = {}
        
        for entity in entities:
            name = entity["name"].lower().strip()
            entity_type = entity.get("type", "UNKNOWN")
            key = f"{name}_{entity_type}"
            
            if key not in entity_map:
                entity_map[key] = {
                    **entity,
                    "source_chunks": [entity["source_chunk"]],
                    "frequency": 1
                }
            else:
                # Update existing entity
                if entity["source_chunk"] not in entity_map[key]["source_chunks"]:
                    entity_map[key]["source_chunks"].append(entity["source_chunk"])
                entity_map[key]["frequency"] += 1
                # Keep more detailed description if available
                if len(entity.get("description", "")) > len(entity_map[key].get("description", "")):
                    entity_map[key]["description"] = entity.get("description", "")
        
        return list(entity_map.values())
