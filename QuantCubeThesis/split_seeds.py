import json
import math
from pathlib import Path

# ── CONFIGURATION ────────────────────────────────────────────────────────────

# Path to your seed files based on your screenshot
INPUT_DIR = Path("data/initial_training_seed_sentences")

# Cross-platform way to locate the user's Downloads folder
# We create a specific subfolder so it doesn't clutter your main Downloads
DOWNLOADS_DIR = Path.home() / "Downloads" / "QuantCube_Seed_Batches"

CHUNK_SIZE = 15

# ── EXECUTION ────────────────────────────────────────────────────────────────

def split_seed_files():
    # 1. Verify input directory exists
    if not INPUT_DIR.exists():
        print(f"❌ Error: Could not find {INPUT_DIR}.")
        print("Make sure you are running this script from the root 'QuantCubeThesis' folder.")
        return

    # 2. Create the destination folder in Downloads
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📁 Destination folder ready: {DOWNLOADS_DIR}\n")

    # 3. Iterate through all JSON files in the seed directory
    for json_file in INPUT_DIR.glob("*.json"):
        with open(json_file, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"⚠️ Skipping {json_file.name}: Not a valid JSON file.")
                continue
        
        # Ensure the JSON contains a list of records
        if not isinstance(data, list):
            print(f"⚠️ Skipping {json_file.name}: Data is not a list.")
            continue
        
        doc_type = json_file.stem  # e.g., 'seed_minutes'
        total_records = len(data)
        
        print(f"⚙️ Processing '{json_file.name}' ({total_records} sentences)...")
        
        # 4. Split into chunks and save
        for i in range(0, total_records, CHUNK_SIZE):
            chunk = data[i : i + CHUNK_SIZE]
            
            # Calculate the current batch number (1 to 10)
            batch_number = (i // CHUNK_SIZE) + 1
            
            # Format filename (e.g., seed_minutes_batch_01.json)
            output_filename = f"{doc_type}_batch_{batch_number:02d}.json"
            output_path = DOWNLOADS_DIR / output_filename
            
            with open(output_path, 'w', encoding='utf-8') as f_out:
                json.dump(chunk, f_out, indent=2)
                
        total_batches = math.ceil(total_records / CHUNK_SIZE)
        print(f"  ✅ Created {total_batches} files.\n")

    print(f"🚀 All done! You can find your split files here: {DOWNLOADS_DIR}")

if __name__ == "__main__":
    split_seed_files()