#!/usr/bin/env python3
"""raglet CLI - Command-line interface for raglet."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

from raglet import RAGlet, RAGletConfig


def build_command(args: argparse.Namespace) -> int:
    """Build knowledge base from workspace files.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, 1 for error)
    """
    workspace = Path(args.workspace)
    if not workspace.exists():
        print(f"Error: Workspace directory not found: {workspace}", file=sys.stderr)
        return 1

    if not workspace.is_dir():
        print(f"Error: Workspace path is not a directory: {workspace}", file=sys.stderr)
        return 1

    # Determine knowledge base path
    kb_path = workspace / args.kb_name if args.kb_name else workspace / ".raglet"

    # Find files to process
    extensions = args.extensions.split(",") if args.extensions else [".txt", ".md", ".py"]
    files = []
    for ext in extensions:
        files.extend(workspace.rglob(f"*{ext}"))

    # Filter out common ignores
    ignore_patterns = args.ignore.split(",") if args.ignore else [
        ".git",
        "__pycache__",
        ".venv",
        "node_modules",
        ".raglet",
    ]
    filtered_files = []
    for file in files:
        if not any(pattern in str(file) for pattern in ignore_patterns):
            filtered_files.append(str(file))

    if not filtered_files:
        print(f"Warning: No files found matching extensions: {extensions}", file=sys.stderr)
        return 1

    print(f"Found {len(filtered_files)} files to process...")
    print(f"Building knowledge base at: {kb_path}")

    try:
        # Create config
        config = RAGletConfig()
        if args.chunk_size:
            config.chunking.size = args.chunk_size
        if args.chunk_overlap:
            config.chunking.overlap = args.chunk_overlap
        if args.model:
            config.embedding.model = args.model

        # Limit files if specified
        files_to_process = filtered_files[: args.max_files] if args.max_files else filtered_files

        # Build RAGlet
        raglet = RAGlet.from_files(files_to_process, config=config)

        # Save
        raglet.save(str(kb_path))

        print(f"✓ Knowledge base built: {len(raglet.chunks)} chunks")
        return 0

    except Exception as e:
        print(f"Error building knowledge base: {e}", file=sys.stderr)
        return 1


def query_command(args: argparse.Namespace) -> int:
    """Query knowledge base.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, 1 for error)
    """
    workspace = Path(args.workspace)
    if not workspace.exists():
        print(f"Error: Workspace directory not found: {workspace}", file=sys.stderr)
        return 1

    # Determine knowledge base path
    kb_path = workspace / args.kb_name if args.kb_name else workspace / ".raglet"

    if not kb_path.exists():
        print(
            f"Error: Knowledge base not found at {kb_path}. Run 'raglet build' first.",
            file=sys.stderr,
        )
        return 1

    try:
        # Load RAGlet
        raglet = RAGlet.load(str(kb_path))

        # Search
        results = raglet.search(args.query, top_k=args.top_k)

        if not results:
            print("No results found.")
            return 0

        # Print results
        print(f"\nFound {len(results)} results:\n")
        for i, result in enumerate(results, 1):
            score = result.score if result.score is not None else 0.0
            print(f"{i}. [{score:.3f}] {result.source}:{result.index}")
            print(f"   {result.text[:200]}...")
            if args.show_full:
                print(f"   Full text: {result.text}\n")
            else:
                print()

        return 0

    except Exception as e:
        print(f"Error querying knowledge base: {e}", file=sys.stderr)
        return 1


def add_command(args: argparse.Namespace) -> int:
    """Add files to existing knowledge base.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, 1 for error)
    """
    workspace = Path(args.workspace)
    if not workspace.exists():
        print(f"Error: Workspace directory not found: {workspace}", file=sys.stderr)
        return 1

    # Determine knowledge base path
    kb_path = workspace / args.kb_name if args.kb_name else workspace / ".raglet"

    if not kb_path.exists():
        print(
            f"Error: Knowledge base not found at {kb_path}. Run 'raglet build' first.",
            file=sys.stderr,
        )
        return 1

    # Resolve file paths
    files = []
    for file_arg in args.files:
        file_path = Path(file_arg)
        if not file_path.is_absolute():
            file_path = workspace / file_path
        if file_path.exists():
            files.append(str(file_path))
        else:
            print(f"Warning: File not found: {file_path}", file=sys.stderr)

    if not files:
        print("Error: No valid files to add.", file=sys.stderr)
        return 1

    try:
        # Load existing RAGlet
        raglet = RAGlet.load(str(kb_path))

        # Add files
        raglet.add_files(files)

        # Save incrementally
        raglet.save(str(kb_path), incremental=True)

        print(f"✓ Added {len(files)} files to knowledge base")
        print(f"  Total chunks: {len(raglet.chunks)}")
        return 0

    except Exception as e:
        print(f"Error adding files: {e}", file=sys.stderr)
        return 1


def export_command(args: argparse.Namespace) -> int:
    """Export knowledge base to zip file.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, 1 for error)
    """
    workspace = Path(args.workspace)
    if not workspace.exists():
        print(f"Error: Workspace directory not found: {workspace}", file=sys.stderr)
        return 1

    # Determine knowledge base path
    kb_path = workspace / args.kb_name if args.kb_name else workspace / ".raglet"

    if not kb_path.exists():
        print(
            f"Error: Knowledge base not found at {kb_path}. Run 'raglet build' first.",
            file=sys.stderr,
        )
        return 1

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = workspace / f"{args.kb_name or 'knowledge'}.zip"

    try:
        # Load RAGlet
        raglet = RAGlet.load(str(kb_path))

        # Export to zip
        import zipfile

        with zipfile.ZipFile(str(output_path), "w", zipfile.ZIP_DEFLATED) as zipf:
            for file_name in ["config.json", "chunks.json", "embeddings.npy", "metadata.json"]:
                file_path = kb_path / file_name
                if file_path.exists():
                    zipf.write(file_path, file_name)

        print(f"✓ Exported knowledge base to: {output_path}")
        return 0

    except Exception as e:
        print(f"Error exporting knowledge base: {e}", file=sys.stderr)
        return 1


def chat_command(args: argparse.Namespace) -> int:
    """Query knowledge base with Claude API.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, 1 for error)
    """
    workspace = Path(args.workspace)
    if not workspace.exists():
        print(f"Error: Workspace directory not found: {workspace}", file=sys.stderr)
        return 1

    # Determine knowledge base path
    kb_path = workspace / args.kb_name if args.kb_name else workspace / ".raglet"

    # Check if knowledge base exists
    if not kb_path.exists():
        print(
            f"Warning: Knowledge base not found at {kb_path}.",
            file=sys.stderr,
        )
        print(
            f"Creating new knowledge base. Use 'raglet build' to create from files.",
            file=sys.stderr,
        )
        # Create empty knowledge base
        config = RAGletConfig()
        raglet = RAGlet(chunks=[], config=config)
        raglet.save(str(kb_path))
        print(f"Created empty knowledge base at {kb_path}")

    try:
        # Load RAGlet
        raglet = RAGlet.load(str(kb_path))

        # Search for relevant chunks
        relevant_chunks = raglet.search(args.query, top_k=args.top_k)

        if not relevant_chunks:
            print("No relevant context found in knowledge base.")
            print("Consider adding files with 'raglet build' or 'raglet add'.")
            context_text = ""
        else:
            # Build context from chunks
            context_text = "\n\n".join(
                [
                    f"[Source: {chunk.source}]\n{chunk.text}"
                    for chunk in relevant_chunks
                ]
            )

        # Get API key
        api_key = args.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print(
                "Error: Anthropic API key required.",
                file=sys.stderr,
            )
            print(
                "Set ANTHROPIC_API_KEY environment variable or use --api-key flag.",
                file=sys.stderr,
            )
            return 1

        # Import Anthropic
        try:
            from anthropic import Anthropic
        except ImportError:
            print(
                "Error: anthropic package not installed.",
                file=sys.stderr,
            )
            print(
                "Install with: pip install anthropic",
                file=sys.stderr,
            )
            return 1

        # Create client and generate response
        client = Anthropic(api_key=api_key)

        # Build prompt
        if context_text:
            prompt = f"""Based on the following context from the knowledge base, answer the user's question.

Context:
{context_text}

User Question: {args.query}

Provide a helpful answer based on the context provided."""
        else:
            prompt = f"""Answer the user's question.

User Question: {args.query}

Provide a helpful answer."""

        # Generate response
        response = client.messages.create(
            model=args.model,
            max_tokens=args.max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        # Print response
        print("\n" + "=" * 80)
        print("Response:")
        print("=" * 80)
        print(response.content[0].text)
        print()

        if relevant_chunks:
            print(f"\n(Used {len(relevant_chunks)} chunks from knowledge base)")
            print("Sources:")
            for i, chunk in enumerate(relevant_chunks, 1):
                score = chunk.score if chunk.score is not None else 0.0
                print(f"  {i}. [{score:.3f}] {chunk.source}")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


def inspect_command(args: argparse.Namespace) -> int:
    """Inspect knowledge base.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, 1 for error)
    """
    workspace = Path(args.workspace)
    if not workspace.exists():
        print(f"Error: Workspace directory not found: {workspace}", file=sys.stderr)
        return 1

    # Determine knowledge base path
    kb_path = workspace / args.kb_name if args.kb_name else workspace / ".raglet"

    if not kb_path.exists():
        print(
            f"Error: Knowledge base not found at {kb_path}. Run 'raglet build' first.",
            file=sys.stderr,
        )
        return 1

    try:
        # Load RAGlet
        raglet = RAGlet.load(str(kb_path))

        # Print information
        print(f"Knowledge Base: {kb_path}\n")
        print(f"Chunks: {len(raglet.chunks)}")
        print(f"Embedding Model: {raglet.config.embedding.model}")
        print(f"Embedding Dimension: {raglet.embeddings.shape[1] if len(raglet.embeddings) > 0 else 0}")
        print(f"Chunk Size: {raglet.config.chunking.size}")
        print(f"Chunk Overlap: {raglet.config.chunking.overlap}")

        # Show sources
        sources = set(chunk.source for chunk in raglet.chunks)
        print(f"\nSources ({len(sources)}):")
        for source in sorted(sources)[:10]:
            count = sum(1 for chunk in raglet.chunks if chunk.source == source)
            print(f"  {source}: {count} chunks")
        if len(sources) > 10:
            print(f"  ... and {len(sources) - 10} more")

        return 0

    except Exception as e:
        print(f"Error inspecting knowledge base: {e}", file=sys.stderr)
        return 1


def main() -> int:
    """Main CLI entry point.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    parser = argparse.ArgumentParser(
        description="raglet - Portable memory for small text corpora",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build knowledge base from workspace
  raglet --workspace /path/to/project build

  # Query knowledge base
  raglet --workspace /path/to/project query "what is Python?"

  # Add files incrementally
  raglet --workspace /path/to/project add file1.txt file2.md

  # Export to zip
  raglet --workspace /path/to/project export --output knowledge.zip

  # Inspect knowledge base
  raglet --workspace /path/to/project inspect

  # Chat with Claude API (uses raglet context)
  raglet --workspace /path/to/project chat "explain Python" --top-k 5
        """,
    )

    # Global arguments
    parser.add_argument(
        "--workspace",
        type=str,
        required=True,
        help="Workspace directory path",
    )
    parser.add_argument(
        "--kb-name",
        type=str,
        default=None,
        help="Knowledge base name (default: .raglet)",
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Build command
    build_parser = subparsers.add_parser("build", help="Build knowledge base from workspace")
    build_parser.add_argument(
        "--extensions",
        type=str,
        default=".txt,.md,.py",
        help="File extensions to process (comma-separated, default: .txt,.md,.py)",
    )
    build_parser.add_argument(
        "--ignore",
        type=str,
        default=".git,__pycache__,.venv,node_modules,.raglet",
        help="Patterns to ignore (comma-separated)",
    )
    build_parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of files to process (default: all)",
    )
    build_parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Chunk size (default: 512)",
    )
    build_parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="Chunk overlap (default: 50)",
    )
    build_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Embedding model (default: all-MiniLM-L6-v2)",
    )

    # Query command
    query_parser = subparsers.add_parser("query", help="Query knowledge base")
    query_parser.add_argument("query", type=str, help="Search query")
    query_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results (default: 5)",
    )
    query_parser.add_argument(
        "--show-full",
        action="store_true",
        help="Show full text of results",
    )

    # Add command
    add_parser = subparsers.add_parser("add", help="Add files to knowledge base")
    add_parser.add_argument("files", nargs="+", help="Files to add")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export knowledge base to zip")
    export_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output zip file path (default: knowledge.zip)",
    )

    # Inspect command
    subparsers.add_parser("inspect", help="Inspect knowledge base")

    # Chat command (LLM-powered query)
    chat_parser = subparsers.add_parser(
        "chat",
        help="Query knowledge base with Claude API (requires ANTHROPIC_API_KEY)",
    )
    chat_parser.add_argument("query", type=str, help="Search query")
    chat_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of context chunks to use (default: 5)",
    )
    chat_parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)",
    )
    chat_parser.add_argument(
        "--model",
        type=str,
        default="claude-opus-4-6",
        help="Claude model to use (default: claude-opus-4-6)",
    )
    chat_parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum tokens in response (default: 1024)",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Route to command handler
    if args.command == "build":
        return build_command(args)
    elif args.command == "query":
        return query_command(args)
    elif args.command == "add":
        return add_command(args)
    elif args.command == "export":
        return export_command(args)
    elif args.command == "inspect":
        return inspect_command(args)
    elif args.command == "chat":
        return chat_command(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
