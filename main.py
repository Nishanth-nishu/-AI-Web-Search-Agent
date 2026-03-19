#!/usr/bin/env python3
"""
AI Agent System — CLI Entry Point

Two subcommands:
  - search: Web search agent (Challenge A)
  - pdf:    PDF RAG agent (Challenge B)

Usage:
  python main.py search "What are the latest specs in MacBook this year?"
  python main.py pdf --file document.pdf --query "What methodology was used?"
  python main.py pdf --file document.pdf --summarize
"""

import argparse
import logging
import sys
import os

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def setup_logging(level: str = "INFO"):
    """Configure logging with Rich formatting if available."""
    try:
        from rich.logging import RichHandler
        logging.basicConfig(
            level=getattr(logging, level),
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(rich_tracebacks=True)],
        )
    except ImportError:
        logging.basicConfig(
            level=getattr(logging, level),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )


def run_search(args):
    """Execute the web search agent."""
    from agent.web_search_agent import WebSearchAgent

    agent = WebSearchAgent()
    response = agent.answer(args.query)
    print("\n" + "="*60)
    print(agent.format_response(response))
    print("="*60)


def run_pdf(args):
    """Execute the PDF RAG agent."""
    from agent.pdf_rag_agent import PDFRAGAgent

    agent = PDFRAGAgent()

    # Ingest the PDF
    print(f"\nIngesting PDF: {args.file}")
    try:
        stats = agent.ingest(args.file)
        print(f"  ✓ Pages: {stats['total_pages']}")
        print(f"  ✓ Chunks: {stats['total_chunks']}")
        print(f"  ✓ Tokens: {stats['total_tokens']}")
    except FileNotFoundError as e:
        print(f"\n  ✗ Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"\n  ✗ Error: {e}")
        sys.exit(1)

    # Summarize or answer questions
    if args.summarize:
        print("\nGenerating summary...")
        response = agent.summarize()
        print("\n" + "="*60)
        print(agent.format_response(response))
        print("="*60)

    if args.query:
        print(f"\nAnswering question: {args.query}")
        response = agent.query(args.query)
        print("\n" + "="*60)
        print(agent.format_response(response))
        print("="*60)

    # Interactive mode if neither flag given
    if not args.summarize and not args.query:
        print("\nEntering interactive mode. Type 'quit' to exit.")
        print("Commands: 'summarize' for summary, or type a question.\n")
        while True:
            try:
                user_input = input("📄 Ask > ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ("quit", "exit", "q"):
                    break
                if user_input.lower() == "summarize":
                    response = agent.summarize()
                else:
                    response = agent.query(user_input)
                print("\n" + agent.format_response(response) + "\n")
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AI Agent System — Web Search & PDF RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py search "What are the latest specs in MacBook this year?"
  python main.py pdf --file report.pdf --summarize
  python main.py pdf --file report.pdf --query "What methodology was used?"
  python main.py pdf --file report.pdf  # Interactive mode
        """,
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Agent to use")

    # ── Web Search subcommand ──
    search_parser = subparsers.add_parser(
        "search", help="Search the web and answer questions"
    )
    search_parser.add_argument(
        "query", type=str, help="Natural language question"
    )

    # ── PDF RAG subcommand ──
    pdf_parser = subparsers.add_parser(
        "pdf", help="Summarize or ask questions about a PDF"
    )
    pdf_parser.add_argument(
        "--file", "-f", type=str, required=True,
        help="Path to the PDF file",
    )
    pdf_parser.add_argument(
        "--query", "-q", type=str, default=None,
        help="Question to ask about the PDF",
    )
    pdf_parser.add_argument(
        "--summarize", "-s", action="store_true",
        help="Generate a summary of the PDF",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    setup_logging(args.log_level)

    if args.command == "search":
        run_search(args)
    elif args.command == "pdf":
        run_pdf(args)


if __name__ == "__main__":
    main()
