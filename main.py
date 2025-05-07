import warnings

warnings.filterwarnings("ignore")

import argparse
import os
import logging

from tqdm import tqdm

from embed_providers import get_embedder
from utils import load_markdown_files, save_to_json, check_api_health, upload_embeddings

# Configure logging
def configure_logging(verbose):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')

    # Suppress specific loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

def main():
    # Argument parser
    parser = argparse.ArgumentParser(
        description="Generate embeddings from markdown (.md + .mdx) files."
    )
    parser.add_argument("--provider", required=True, choices=["openai"],help="Embedding provider to use.")
    parser.add_argument("--model", required=True, help="Model name to use for the embedding provider.")
    parser.add_argument("--api-key", help="API key for the embedding provider.")
    parser.add_argument("--input-path", required=True, help="Path to the markdown docs folder.")
    parser.add_argument("--output-path", default="embeddings.json", help="Output file path for embeddings JSON")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--backend-api-url", help="URL of the API to upload embeddings.")
    parser.add_argument("--backend-api-key", help="API key for the API.")

    args = parser.parse_args()

    # Configure logging
    configure_logging(args.verbose)

    logging.debug("Starting the embedding generation process.")

    # Validate arguments
    if args.provider == "openai":
        if not args.api_key:
            parser.error("--api-key is required for OpenAI")
        embedder = get_embedder("openai", api_key=args.api_key, model=args.model)

    # Validate the input path
    if not os.path.isdir(args.input_path):
        parser.error(f"Input path '{args.input_path}' does not exist or is not a directory.")

    # Validate if the backend API is reachable
    if args.backend_api_url:
        logging.info("Checking API health...")
        if not check_api_health(args.backend_api_url, args.backend_api_key):
            logging.error("❌ API is not reachable. Exiting.")
            exit(1)

    # Validate the embedder
    try:
        logging.debug("Validating the embedder.")
        embedder.validate()
    except (ValueError, RuntimeError) as e:
        logging.error(e)
        exit(1)

    # Load documentation files
    logging.info("Loading markdown files.")
    docs = load_markdown_files(args.input_path)

    # Generate embeddings
    logging.info("🔄 Generating embeddings...")
    embeddings = []
    for doc in tqdm(docs, desc="Processing documents"):
        logging.debug(f"Processing document: {doc}")
        embeddings.append(embedder.embed_documents([doc])[0])

    # Upload or save embeddings
    if args.backend_api_url and args.backend_api_key:
        logging.info("Uploading embeddings to API.")
        upload_embeddings(args.backend_api_url, args.backend_api_key, embeddings)
    else:
        logging.info("Saving embeddings to JSON.")
        save_to_json(embeddings, args.output_path)
        logging.info(f"✅ Embeddings saved to {args.output_path}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.warning("Operation cancelled by user.")