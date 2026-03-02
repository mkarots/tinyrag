"""TinyRAG main class."""

from typing import List, Optional

from tinyrag.core.chunk import Chunk
from tinyrag.processing.interfaces import DocumentExtractor, Chunker
from tinyrag.config.config import TinyRAGConfig


class TinyRAG:
    """Main RAG orchestrator class."""
    
    def __init__(
        self,
        chunks: List[Chunk],
        config: TinyRAGConfig
    ):
        """Initialize TinyRAG.
        
        Args:
            chunks: List of chunks
            config: Configuration
        """
        self.chunks = chunks
        self.config = config
    
    @classmethod
    def from_files(
        cls,
        files: List[str],
        document_extractor: Optional[DocumentExtractor] = None,
        chunker: Optional[Chunker] = None,
        config: Optional[TinyRAGConfig] = None,
    ) -> "TinyRAG":
        """Create TinyRAG from files.
        
        Args:
            files: List of file paths
            document_extractor: Optional document extractor (uses factory if None)
            chunker: Optional chunker (creates default if None)
            config: Optional configuration (uses defaults if None)
            
        Returns:
            TinyRAG instance
        """
        if config is None:
            config = TinyRAGConfig()
        
        config.validate()
        
        # This is a skeleton - will be completed in Milestone 2
        # For now, just extract and chunk
        from tinyrag.processing.extractor_factory import create_extractor
        from tinyrag.processing.chunker import SentenceAwareChunker
        
        all_chunks = []
        
        for file_path in files:
            if document_extractor is None:
                extractor = create_extractor(file_path)
            else:
                extractor = document_extractor
            
            text = extractor.extract(file_path)
            
            if chunker is None:
                chunker_instance = SentenceAwareChunker(config.chunking)
            else:
                chunker_instance = chunker
            
            chunks = chunker_instance.chunk(text, metadata={"source": file_path})
            all_chunks.extend(chunks)
        
        return cls(chunks=all_chunks, config=config)
    
    def get_all_chunks(self) -> List[Chunk]:
        """Get all chunks.
        
        Returns:
            List of all chunks
        """
        return self.chunks
