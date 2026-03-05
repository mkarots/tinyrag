"""Unit tests for CLI command functions."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from raglet.cli import (
    add_command,
    build_command,
    export_command,
    inspect_command,
    query_command,
)
from raglet.core.chunk import Chunk


@pytest.mark.unit
class TestCLICommands:
    """Test CLI command functions."""

    def test_build_command_creates_directory(self):
        """Test build_command creates knowledge base directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            kb_path = workspace / ".raglet"

            # Create test files
            (workspace / "test.txt").write_text("Test content")

            # Mock args
            args = MagicMock()
            args.workspace = str(workspace)
            args.kb_name = None
            args.extensions = ".txt,.md"
            args.ignore = ".git,__pycache__"
            args.max_files = None
            args.chunk_size = None
            args.chunk_overlap = None
            args.model = None

            # Run build command
            result = build_command(args)

            assert result == 0
            assert kb_path.exists()
            assert (kb_path / "chunks.json").exists()

    def test_build_command_handles_no_files(self):
        """Test build_command handles case with no matching files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)

            # Mock args
            args = MagicMock()
            args.workspace = str(workspace)
            args.kb_name = None
            args.extensions = ".xyz"  # No files match
            args.ignore = ""
            args.max_files = None
            args.chunk_size = None
            args.chunk_overlap = None
            args.model = None

            # Run build command
            result = build_command(args)

            assert result == 1  # Should fail with no files

    def test_query_command_loads_and_searches(self):
        """Test query_command loads knowledge base and searches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            kb_path = workspace / ".raglet"

            # Create knowledge base
            chunks = [
                Chunk(text="Python programming", source="test.txt", index=0),
            ]
            from raglet import RAGlet, RAGletConfig

            config = RAGletConfig()
            raglet = RAGlet.from_files([], config=config)
            raglet.chunks = chunks
            raglet.embeddings = raglet.embedding_generator.generate(chunks)
            raglet.vector_store.add_vectors(raglet.embeddings, chunks)
            raglet.save(str(kb_path))

            # Mock args
            args = MagicMock()
            args.workspace = str(workspace)
            args.kb_name = None
            args.query = "Python"
            args.top_k = 5
            args.show_full = False

            # Run query command
            result = query_command(args)

            assert result == 0

    def test_query_command_handles_missing_kb(self):
        """Test query_command handles missing knowledge base."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)

            # Mock args
            args = MagicMock()
            args.workspace = str(workspace)
            args.kb_name = None
            args.query = "test"
            args.top_k = 5
            args.show_full = False

            # Run query command (should fail)
            result = query_command(args)

            assert result == 1  # Should fail with missing KB

    def test_add_command_adds_files(self):
        """Test add_command adds files to knowledge base."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            kb_path = workspace / ".raglet"

            # Create initial knowledge base
            chunks = [
                Chunk(text="Initial", source="initial.txt", index=0),
            ]
            from raglet import RAGlet, RAGletConfig

            config = RAGletConfig()
            raglet = RAGlet.from_files([], config=config)
            raglet.chunks = chunks
            raglet.embeddings = raglet.embedding_generator.generate(chunks)
            raglet.vector_store.add_vectors(raglet.embeddings, chunks)
            raglet.save(str(kb_path))

            # Create new file
            new_file = workspace / "new.txt"
            new_file.write_text("New content")

            # Mock args
            args = MagicMock()
            args.workspace = str(workspace)
            args.kb_name = None
            args.files = ["new.txt"]

            # Run add command
            result = add_command(args)

            assert result == 0

            # Verify file added
            loaded = RAGlet.load(str(kb_path))
            assert len(loaded.chunks) > len(chunks)

    def test_add_command_handles_missing_kb(self):
        """Test add_command handles missing knowledge base."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)

            # Mock args
            args = MagicMock()
            args.workspace = str(workspace)
            args.kb_name = None
            args.files = ["test.txt"]

            # Run add command (should fail)
            result = add_command(args)

            assert result == 1  # Should fail with missing KB

    def test_export_command_creates_zip(self):
        """Test export_command creates zip file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            kb_path = workspace / ".raglet"
            zip_path = workspace / "export.zip"

            # Create knowledge base
            chunks = [
                Chunk(text="Test", source="test.txt", index=0),
            ]
            from raglet import RAGlet, RAGletConfig

            config = RAGletConfig()
            raglet = RAGlet.from_files([], config=config)
            raglet.chunks = chunks
            raglet.embeddings = raglet.embedding_generator.generate(chunks)
            raglet.vector_store.add_vectors(raglet.embeddings, chunks)
            raglet.save(str(kb_path))

            # Mock args
            args = MagicMock()
            args.workspace = str(workspace)
            args.kb_name = None
            args.output = str(zip_path)

            # Run export command
            result = export_command(args)

            assert result == 0
            assert zip_path.exists()

    def test_inspect_command_shows_info(self):
        """Test inspect_command shows knowledge base information."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            kb_path = workspace / ".raglet"

            # Create knowledge base
            chunks = [
                Chunk(text="Python", source="test.txt", index=0),
            ]
            from raglet import RAGlet, RAGletConfig

            config = RAGletConfig()
            raglet = RAGlet.from_files([], config=config)
            raglet.chunks = chunks
            raglet.embeddings = raglet.embedding_generator.generate(chunks)
            raglet.vector_store.add_vectors(raglet.embeddings, chunks)
            raglet.save(str(kb_path))

            # Mock args
            args = MagicMock()
            args.workspace = str(workspace)
            args.kb_name = None

            # Run inspect command
            result = inspect_command(args)

            assert result == 0

    def test_inspect_command_handles_missing_kb(self):
        """Test inspect_command handles missing knowledge base."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)

            # Mock args
            args = MagicMock()
            args.workspace = str(workspace)
            args.kb_name = None

            # Run inspect command (should fail)
            result = inspect_command(args)

            assert result == 1  # Should fail with missing KB

    def test_build_command_with_custom_kb_name(self):
        """Test build_command with custom knowledge base name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            kb_path = workspace / "custom_kb"

            # Create test files
            (workspace / "test.txt").write_text("Test content")

            # Mock args
            args = MagicMock()
            args.workspace = str(workspace)
            args.kb_name = "custom_kb"
            args.extensions = ".txt"
            args.ignore = ""
            args.max_files = None
            args.chunk_size = None
            args.chunk_overlap = None
            args.model = None

            # Run build command
            result = build_command(args)

            assert result == 0
            assert kb_path.exists()

    def test_query_command_with_custom_kb_name(self):
        """Test query_command with custom knowledge base name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            kb_path = workspace / "custom_kb"

            # Create knowledge base
            chunks = [
                Chunk(text="Python", source="test.txt", index=0),
            ]
            from raglet import RAGlet, RAGletConfig

            config = RAGletConfig()
            raglet = RAGlet.from_files([], config=config)
            raglet.chunks = chunks
            raglet.embeddings = raglet.embedding_generator.generate(chunks)
            raglet.vector_store.add_vectors(raglet.embeddings, chunks)
            raglet.save(str(kb_path))

            # Mock args
            args = MagicMock()
            args.workspace = str(workspace)
            args.kb_name = "custom_kb"
            args.query = "Python"
            args.top_k = 5
            args.show_full = False

            # Run query command
            result = query_command(args)

            assert result == 0
