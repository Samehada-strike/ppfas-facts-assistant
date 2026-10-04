import re
from pathlib import Path


# === CONFIGURATION ===
BASE_DIR = Path(r"E:\ML Projects\RAG_Chatbot_PP_MF")

INPUT_DIR = BASE_DIR / "project_files" / "data" / "raw" / "Markdown"
OUTPUT_DIR = BASE_DIR / "project_files" / "data" / "raw" / "Markdown_cleaned"


def remove_urls_from_markdown(content: str) -> str:
    """
    Removes URLs from markdown content while preserving link text and image alt text.
    """
    # Step 1: Replace markdown links [text](url) with just text
    content = re.sub(r'\[([^\]]+)\]\(https?://[^\)]+\)', r'\1', content)

    # Step 2: Replace image markdown links ![alt](url) with alt text
    content = re.sub(r'!\[([^\]]*)\]\(https?://[^\)]+\)', r'\1', content)

    # Step 3: Remove remaining raw URLs
    content = re.sub(r'https?://[^\s\)]+', '', content)

    # Step 4: Cleanup formatting
    content = re.sub(r'  +', ' ', content)          # multiple spaces -> single
    content = re.sub(r'\n\n\n+', '\n\n', content)   # multiple blank lines -> double

    return content


def main():
    print("Markdown cleaning job started...\n")

    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    md_files = list(INPUT_DIR.glob("*.md"))

    if not md_files:
        print("No .md files found in input directory.")
        return

    print(f"Found {len(md_files)} markdown files.")
    print("-" * 50)

    for md_file in md_files:
        print(f"Processing: {md_file.name}")

        # Read original content
        original_text = md_file.read_text(encoding="utf-8")

        # Clean content
        cleaned_text = remove_urls_from_markdown(original_text)

        # Write cleaned version
        output_file = OUTPUT_DIR / md_file.name
        output_file.write_text(cleaned_text, encoding="utf-8")

        print(f"✓ Saved cleaned file: {output_file}")

    print("\n✅ All files cleaned and saved successfully.")


if __name__ == "__main__":
    main()