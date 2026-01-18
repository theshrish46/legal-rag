import pandas as pd
from pypdf import PdfReader
import os
import sys

# --- FIX 1: CORRECT PATH SETUP ---
# Get the folder where THIS script lives (evals/)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent folder (legal-rag/)
parent_dir = os.path.dirname(current_dir)

# Add parent folder to system path so we can import 'src'
sys.path.append(parent_dir)

# --- FIX 2: ABSOLUTE IMPORT (No '..') ---
from src.prompts.dataset_generator_prompt import get_generator_chain


def generate_data():
    # --- FIX 3: POINT TO THE ACTUAL DATA FOLDER ---
    # Construct the path: legal-rag/DATA/UPLOADED/filename.pdf
    filename = "ChinaRealEstateInformationCorp_20090929_F-1_EX-10.32_4771615_EX-10.32_Content License Agreement.pdf"

    # Use os.path.join to handle Windows/Mac slashes automatically
    pdf_path = os.path.join(parent_dir, "DATA", "UPLOADED", filename)

    # Safety check
    if not os.path.exists(pdf_path):
        print(f"❌ CRITICAL ERROR: File not found at: {pdf_path}")
        print("Check if the file is actually in the DATA/UPLOADED folder.")
        return None

    pdf_reader = PdfReader(pdf_path)
    text = ""
    # Only read first 5 pages to save tokens/money
    for page in pdf_reader.pages[:5]:
        text += page.extract_text() or ""

    text = " ".join(text.replace("\n", " ").split()).lower().strip()
    return text


if __name__ == "__main__":
    text = generate_data()

    if text:
        print("Text extracted. Generating questions...")

        # NOTE: Assuming get_generator_chain returns the DATA directly.
        # If it returns a Chain object, change this line to:
        # data = get_generator_chain().invoke({"text": text})
        data = get_generator_chain(text=text)

        df = pd.DataFrame(data)

        # Save CSV in the current folder (evals/)
        output_path = os.path.join(current_dir, "generated_data.csv")
        df.to_csv(output_path, index=False)

        print(f"✅ Success! Saved to {output_path}")
        print(df.head())
    else:
        print("Failed to read PDF.")
