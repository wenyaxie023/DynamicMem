from typing import List, Dict, Optional, Any, Tuple
import uuid
from datetime import datetime
from .llm_controller import LLMController
from .llm_text_parsers import (
    ANALYZE_CONTENT_PROMPT,
    EVOLUTION_DECISION_PROMPT,
    FOCUSED_KEYWORDS_PROMPT,
    STRENGTHEN_DETAILS_PROMPT,
    UPDATE_NEIGHBORS_PROMPT,
    _heuristic_context,
    _heuristic_keywords,
    _parse_list_items,
    parse_analyze_content,
    parse_evolution_decision,
    parse_strengthen_details,
    parse_update_neighbors,
    validate_analysis_result,
)
from .retrievers import SimpleEmbeddingRetriever
import logging
import os
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

class MemoryNote:
    """A memory note that represents a single unit of information in the memory system.
    
    This class encapsulates all metadata associated with a memory, including:
    - Core content and identifiers
    - Temporal information (creation and access times)
    - Semantic metadata (keywords, context, tags)
    - Relationship data (links to other memories)
    - Usage statistics (retrieval count)
    - Evolution tracking (history of changes)
    """
    
    def __init__(self, 
                 content: str,
                 id: Optional[str] = None,
                 keywords: Optional[List[str]] = None,
                 links: Optional[Dict] = None,
                 retrieval_count: Optional[int] = None,
                 timestamp: Optional[str] = None,
                 last_accessed: Optional[str] = None,
                 context: Optional[str] = None,
                 evolution_history: Optional[List] = None,
                 category: Optional[str] = None,
                 tags: Optional[List[str]] = None):
        """Initialize a new memory note with its associated metadata.
        
        Args:
            content (str): The main text content of the memory
            id (Optional[str]): Unique identifier for the memory. If None, a UUID will be generated
            keywords (Optional[List[str]]): Key terms extracted from the content
            links (Optional[Dict]): References to related memories
            retrieval_count (Optional[int]): Number of times this memory has been accessed
            timestamp (Optional[str]): Creation time in format YYYYMMDDHHMM
            last_accessed (Optional[str]): Last access time in format YYYYMMDDHHMM
            context (Optional[str]): The broader context or domain of the memory
            evolution_history (Optional[List]): Record of how the memory has evolved
            category (Optional[str]): Classification category
            tags (Optional[List[str]]): Additional classification tags
        """
        # Core content and ID
        self.content = content
        self.id = id or str(uuid.uuid4())
        
        # Semantic metadata
        self.keywords = keywords or []
        self.links = links or []
        self.context = context or "General"
        self.category = category or "Uncategorized"
        self.tags = tags or []
        
        # Temporal information
        current_time = datetime.now().strftime("%Y%m%d%H%M")
        self.timestamp = timestamp or current_time
        self.last_accessed = last_accessed or current_time
        
        # Usage and evolution data
        self.retrieval_count = retrieval_count or 0
        self.evolution_history = evolution_history or []

class AgenticMemorySystem:
    """Core memory system that manages memory notes and their evolution.
    
    This system provides:
    - Memory creation, retrieval, update, and deletion
    - Content analysis and metadata extraction
    - Memory evolution and relationship management
    - Hybrid search capabilities
    """
    
    def __init__(self, 
                embedding_model_name: str = 'all-MiniLM-L6-v2',
                embedding_backend: str = "sentence-transformers",
                embedding_api_key: Optional[str] = None,
                embedding_api_base_url: Optional[str] = None,
                
                llm_controller_backend: str = "openai",
                llm_controller_model_name: str = "gpt-4o-mini",
                llm_controller_api_key: Optional[str] = None,
                llm_controller_api_base_url: Optional[str] = None,
                
                evo_threshold: int = 100,
                collection_name: str = "memories",
                retriever_directory: Optional[str] = None,
                reset_collection: bool = True):  
        """Initialize the memory system.
        
        Args:
            embedding_model_name: Name of the embedding model
            embedding_backend: Embedding backend (sentence-transformers/openai)
            embedding_api_key: API key for embedding provider (optional)
            embedding_api_base_url: Base URL for embedding provider (optional)
            collection_name: Snapshot collection label retained for TCE compatibility
            llm_controller_backend: LLM backend to use (openai/ollama)
            llm_controller_model_name: Name of the LLM model
            llm_controller_api_key: API key for the LLM service (optional)
            llm_controller_api_base_url: Base URL for the LLM provider (optional)
            evo_threshold: Number of memories before triggering evolution
        """
        self.memories = {}
        self.embedding_model_name = embedding_model_name
        self.embedding_backend = embedding_backend
        self.collection_name = collection_name
        self.retriever_directory = retriever_directory
        self.embedding_api_key = embedding_api_key
        self.embedding_api_base_url = embedding_api_base_url
        self.llm_controller_backend = llm_controller_backend
        self.llm_controller_model_name = llm_controller_model_name
        self.llm_controller_api_key = llm_controller_api_key or embedding_api_key
        self.llm_controller_api_base_url = llm_controller_api_base_url or embedding_api_base_url
        self.retriever = SimpleEmbeddingRetriever(
            model_name=self.embedding_model_name,
            embedding_backend=self.embedding_backend,
            openai_api_key=self.embedding_api_key,
            openai_api_base=self.embedding_api_base_url,
            directory=self.retriever_directory,
            extend=not reset_collection,
            collection_name=self.collection_name,
        )
        
        # Initialize LLM controller
        self.llm_controller = LLMController(
            self.llm_controller_backend,
            self.llm_controller_model_name,
            self.llm_controller_api_key,
            self.llm_controller_api_base_url,
        )
        self.evo_cnt = 0
        self.evo_threshold = evo_threshold

    @staticmethod
    def _normalize_link_index(value: Any) -> Optional[int]:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            token = value.strip()
            if not token:
                return None
            try:
                return int(token)
            except ValueError:
                return None
        return None

    def _ordered_memory_items(self) -> List[Tuple[str, MemoryNote]]:
        return list(self.memories.items())

    @staticmethod
    def _rebuild_retriever_document(memory: MemoryNote) -> str:
        metadata_text = f"{memory.context} {' '.join(memory.keywords)} {' '.join(memory.tags)}"
        return f"{memory.content} , {metadata_text}"

    def save_state(self, path: Path) -> None:
        """Persist in-memory notes to disk for resuming runs."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("wb") as f:
            pickle.dump(self.memories, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, path)

    def load_state(self, path: Path) -> None:
        """Load previously saved in-memory notes."""
        path = Path(path)
        with path.open("rb") as f:
            self.memories = pickle.load(f)

    def rebuild_retriever(self) -> None:
        """Rebuild retriever from in-memory notes to ensure sync."""
        self.retriever = SimpleEmbeddingRetriever(
            model_name=self.embedding_model_name,
            embedding_backend=self.embedding_backend,
            openai_api_key=self.embedding_api_key,
            openai_api_base=self.embedding_api_base_url,
            directory=self.retriever_directory,
            extend=False,
            collection_name=self.collection_name,
        )

        for memory in self.memories.values():
            self.retriever.add_documents([self._rebuild_retriever_document(memory)])
        
    def analyze_content(self, content: str) -> Dict:            
        """Analyze content using LLM to extract semantic metadata.
        
        Uses a language model to understand the content and extract:
        - Keywords: Important terms and concepts
        - Context: Overall domain or theme
        - Tags: Classification categories
        
        Args:
            content (str): The text content to analyze
            
        Returns:
            Dict: Contains extracted metadata with keys:
                - keywords: List[str]
                - context: str
                - tags: List[str]
        """
        prompt = ANALYZE_CONTENT_PROMPT.format(content=content)
        try:
            response = self.llm_controller.llm.get_completion(prompt)
            analysis = parse_analyze_content(response, content)

            if not analysis["keywords"]:
                retry_prompt = FOCUSED_KEYWORDS_PROMPT.format(content=content)
                retry_response = self.llm_controller.llm.get_completion(retry_prompt, temperature=0.3)
                analysis["keywords"] = _parse_list_items(retry_response)

            return validate_analysis_result(analysis, content)
        except Exception as e:
            logger.error("Error analyzing content: %s", e)
            return {
                "keywords": _heuristic_keywords(content),
                "context": _heuristic_context(content) or "General",
                "tags": _heuristic_keywords(content, 3),
            }

    def add_note(self, content: str, time: str = None, **kwargs) -> str:
        """Add a new memory note"""
        # Create MemoryNote without llm_controller
        if time is not None:
            kwargs['timestamp'] = time
        note = MemoryNote(content=content, **kwargs)

        needs_analysis = (
            not note.keywords or
            note.context == "General" or
            not note.tags
        )
        if needs_analysis:
            analysis = self.analyze_content(content)
            if not note.keywords:
                note.keywords = list(analysis.get("keywords") or [])
            if note.context == "General":
                note.context = str(analysis.get("context") or "General")
            if not note.tags:
                note.tags = list(analysis.get("tags") or [])
        
        # Update retriever with all documents
        evo_label, note = self.process_memory(note)
        self.memories[note.id] = note
        
        # Add to retriever with complete metadata
        metadata = {
            "id": note.id,
            "content": note.content,
            "keywords": note.keywords,
            "links": note.links,
            "retrieval_count": note.retrieval_count,
            "timestamp": note.timestamp,
            "last_accessed": note.last_accessed,
            "context": note.context,
            "evolution_history": note.evolution_history,
            "category": note.category,
            "tags": note.tags
        }
        self.retriever.add_document(note.content, metadata, note.id)
        
        if evo_label == True:
            self.evo_cnt += 1
            if self.evo_cnt % self.evo_threshold == 0:
                self.consolidate_memories()
        return note.id
    
    def consolidate_memories(self):
        """Consolidate memories: update retriever with new documents"""
        self.retriever = SimpleEmbeddingRetriever(
            model_name=self.embedding_model_name,
            embedding_backend=self.embedding_backend,
            openai_api_key=self.embedding_api_key,
            openai_api_base=self.embedding_api_base_url,
            directory=self.retriever_directory,
            extend=False,
            collection_name=self.collection_name,
        )
        
        # Re-add all memory documents with their complete metadata
        for memory in self.memories.values():
            self.retriever.add_documents([self._rebuild_retriever_document(memory)])
    
    def find_related_memories(self, query: str, k: int = 5) -> Tuple[str, List[int]]:
        """Find related memories using the original SimpleEmbeddingRetriever path."""
        if not self.memories:
            return "", []
            
        try:
            ordered_items = self._ordered_memory_items()
            ordered_memories = [memory for _, memory in ordered_items]
            indices = self.retriever.search(query, k)

            memory_str = ""
            normalized_indices: List[int] = []
            for memory_index in indices:
                if memory_index < 0 or memory_index >= len(ordered_memories):
                    continue
                memory = ordered_memories[memory_index]
                memory_str += (
                    f"memory index:{memory_index}\ttalk start time:{memory.timestamp}"
                    f"\tmemory content: {memory.content}"
                    f"\tmemory context: {memory.context}"
                    f"\tmemory keywords: {str(memory.keywords)}"
                    f"\tmemory tags: {str(memory.tags)}\n"
                )
                normalized_indices.append(memory_index)
                    
            return memory_str, normalized_indices
        except Exception as e:
            logger.error(f"Error in find_related_memories: {str(e)}")
            return "", []

    def find_related_memories_raw(self, query: str, k: int = 5) -> str:
        """Find related memories using raw note content plus linked neighbors."""
        if not self.memories:
            return ""
            
        primary_indices = self.find_related_memories(query, k=k)[1]
        ordered_memories = [memory for _, memory in self._ordered_memory_items()]
        memory_str = ""

        for primary_idx in primary_indices[:k]:
            if primary_idx < 0 or primary_idx >= len(ordered_memories):
                continue
            memory = ordered_memories[primary_idx]
            memory_str += (
                f"talk start time:{memory.timestamp}\tmemory content: {memory.content}"
                f"\tmemory context: {memory.context}\tmemory keywords: {str(memory.keywords)}"
                f"\tmemory tags: {str(memory.tags)}\n"
            )
            neighbor_count = 0
            for raw_link in memory.links:
                link_idx = self._normalize_link_index(raw_link)
                if link_idx is None or link_idx < 0 or link_idx >= len(ordered_memories):
                    continue
                neighbor = ordered_memories[link_idx]
                memory_str += (
                    f"talk start time:{neighbor.timestamp}\tmemory content: {neighbor.content}"
                    f"\tmemory context: {neighbor.context}\tmemory keywords: {str(neighbor.keywords)}"
                    f"\tmemory tags: {str(neighbor.tags)}\n"
                )
                neighbor_count += 1
                if neighbor_count >= k:
                    break

        return memory_str

    def read(self, memory_id: str) -> Optional[MemoryNote]:
        """Retrieve a memory note by its ID.
        
        Args:
            memory_id (str): ID of the memory to retrieve
            
        Returns:
            MemoryNote if found, None otherwise
        """
        return self.memories.get(memory_id)
    
    def update(self, memory_id: str, **kwargs) -> bool:
        """Update a memory note.
        
        Args:
            memory_id: ID of memory to update
            **kwargs: Fields to update
            
        Returns:
            bool: True if update successful
        """
        if memory_id not in self.memories:
            return False
            
        note = self.memories[memory_id]
        
        # Update fields
        for key, value in kwargs.items():
            if hasattr(note, key):
                setattr(note, key, value)
                
        # Update in retriever
        metadata = {
            "id": note.id,
            "content": note.content,
            "keywords": note.keywords,
            "links": note.links,
            "retrieval_count": note.retrieval_count,
            "timestamp": note.timestamp,
            "last_accessed": note.last_accessed,
            "context": note.context,
            "evolution_history": note.evolution_history,
            "category": note.category,
            "tags": note.tags
        }
        
        # Delete and re-add to update
        self.retriever.delete_document(memory_id)
        self.retriever.add_document(document=note.content, metadata=metadata, doc_id=memory_id)
        
        return True
    
    def delete(self, memory_id: str) -> bool:
        """Delete a memory note by its ID.
        
        Args:
            memory_id (str): ID of the memory to delete
            
        Returns:
            bool: True if memory was deleted, False if not found
        """
        if memory_id in self.memories:
            # Delete from retriever
            self.retriever.delete_document(memory_id)
            # Delete from local storage
            del self.memories[memory_id]
            return True
        return False
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for memories using the original SimpleEmbeddingRetriever backend."""
        search_indices = self.retriever.search(query, k)
        ordered_items = self._ordered_memory_items()
        memories = []
        
        for idx in search_indices:
            if idx < 0 or idx >= len(ordered_items):
                continue
            doc_id, memory = ordered_items[idx]
            memories.append({
                'id': doc_id,
                'content': memory.content,
                'context': memory.context,
                'keywords': memory.keywords,
                'score': 0.0,
            })
        
        return memories[:k]
    
    def process_memory(self, note: MemoryNote) -> Tuple[bool, MemoryNote]:
        """Process a memory note and determine if it should evolve.
        
        Args:
            note: The memory note to process
            
        Returns:
            Tuple[bool, MemoryNote]: (should_evolve, processed_note)
        """
        # For first memory or testing, just return the note without evolution
        if not self.memories:
            return False, note
            
        
        neighbors_text, indices = self.find_related_memories(note.content, k=5)
        if not neighbors_text or not indices:
            return False, note

        try:
            decision_prompt = EVOLUTION_DECISION_PROMPT.format(
                context=note.context,
                content=note.content,
                keywords=note.keywords,
                nearest_neighbors_memories=neighbors_text,
            )
            decision_response = self.llm_controller.llm.get_completion(decision_prompt)
            decision = parse_evolution_decision(decision_response)
            if decision["decision"] == "NO_EVOLUTION":
                return False, note

            should_strengthen = decision["decision"] in ("STRENGTHEN", "STRENGTHEN_AND_UPDATE")
            should_update = decision["decision"] in ("UPDATE_NEIGHBOR", "STRENGTHEN_AND_UPDATE")

            if should_strengthen:
                strengthen_prompt = STRENGTHEN_DETAILS_PROMPT.format(
                    content=note.content,
                    keywords=note.keywords,
                    nearest_neighbors_memories=neighbors_text,
                )
                strengthen_response = self.llm_controller.llm.get_completion(strengthen_prompt)
                strengthen = parse_strengthen_details(strengthen_response)
                suggested_connections: List[int] = []
                for raw_link in strengthen["connections"]:
                    link_idx = self._normalize_link_index(raw_link)
                    if link_idx is None:
                        continue
                    suggested_connections.append(link_idx)
                note.links.extend(suggested_connections)
                if strengthen["tags"]:
                    note.tags = strengthen["tags"]

            if should_update:
                update_prompt = UPDATE_NEIGHBORS_PROMPT.format(
                    content=note.content,
                    context=note.context,
                    nearest_neighbors_memories=neighbors_text,
                    max_neighbor_idx=len(indices) - 1,
                    neighbor_count=len(indices),
                )
                update_response = self.llm_controller.llm.get_completion(update_prompt)
                neighbor_updates = parse_update_neighbors(update_response, len(indices))
                noteslist = [memory for _, memory in self._ordered_memory_items()]
                notes_id = [memory_id for memory_id, _ in self._ordered_memory_items()]

                for i in range(min(len(indices), len(neighbor_updates))):
                    memorytmp_idx = indices[i]
                    if memorytmp_idx < 0 or memorytmp_idx >= len(noteslist):
                        continue
                    update = neighbor_updates[i]
                    notetmp = noteslist[memorytmp_idx]
                    if update["tags"]:
                        notetmp.tags = update["tags"]
                    if update["context"]:
                        notetmp.context = update["context"]
                    self.memories[notes_id[memorytmp_idx]] = notetmp

            return True, note
        except Exception as e:
            logger.error("Evolution failed for note %s: %s — storing without evolution", note.id, e)
            return False, note
