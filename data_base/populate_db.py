import os
import csv
import torch
import psycopg
from dotenv import load_dotenv

load_dotenv()

# Connection setup
DB_CONN_STRING = "host=localhost port=5432 dbname=voxceleb_speaker_id user=postgres password=mysecretpassword"
PT_FILE_PATH = r'c:\Users\Konstantinos\Downloads\voxceleb_embeddings (1).pt'
METADATA_CSV_PATH = os.getenv("METADATA_CSV_PATH")

def load_metadata_lookup(csv_path):
    """
    Reads the VoxCeleb metadata file and builds a high-speed
    lookup dictionary mapped by Speaker ID.
    Auto-detects if the file is comma-separated or tab-separated.
    """
    metadata_lookup = {}
    
    if not csv_path or not os.path.exists(csv_path):
        print(f"Warning: Metadata file not found at {csv_path}. Defaulting to 'unknown'.")
        return metadata_lookup

    with open(csv_path, mode='r', encoding='utf-8') as f:
        # Smart Fix: Peek at the header line to see if it contains tabs or commas
        first_line = f.readline()
        delimiter = '\t' if '\t' in first_line else ','
        f.seek(0)  # Reset file pointer back to the beginning
        
        print(f"🔍 Auto-detected delimiter: {'[TAB]' if delimiter == '\t' else '[COMMA]'}")
        
        reader = csv.DictReader(f, delimiter=delimiter)
        
        for row in reader:
            # Strip potential whitespace from headers and values
            spk_id = row.get('VoxCeleb1 ID', '').strip()
            gender = row.get('Gender', 'unknown').strip().lower()
            nationality = row.get('Nationality', 'unknown').strip()
            
            # Standardize gender strings to match your database constraints
            gender_full = "male" if gender in ["m", "male"] else "female" if gender in ["f", "female"] else "unknown"
            
            if spk_id:
                metadata_lookup[spk_id] = {
                    "gender": gender_full,
                    "nationality": nationality
                }
                
    print(f"📚 Successfully indexed real metadata for {len(metadata_lookup)} speakers.")
    return metadata_lookup


def load_data_to_postgres():
    if not os.path.exists(PT_FILE_PATH):
        print(f"Target file not found at: {PT_FILE_PATH}")
        return

    # Load metadata lookup with auto-detection
    metadata_map = load_metadata_lookup(METADATA_CSV_PATH)

    print("Loading .pt file into Python memory... (this might take a few seconds)")
    data = torch.load(PT_FILE_PATH, map_location=torch.device('cpu'))
    
    embeddings = data['embeddings'].tolist() # Convert PyTorch tensor to Python lists
    speakers = data['speakers']
    processed_files = data['processed_files']
    
    total_records = len(speakers)
    print(f"Found {total_records} records to process.")

    print("Connecting to PostgreSQL...")
    with psycopg.connect(DB_CONN_STRING) as conn:
        with conn.cursor() as cur:
            
            # --- STEP 1: Populate Unique Speakers (Using REAL Metadata) ---
            print("Extracting and inserting unique speakers...")
            unique_speakers = set(speakers)
            
            speaker_rows = []
            for spk_id in unique_speakers:
                # Fetch real data from your map, fallback safely if missing
                spk_meta = metadata_map.get(spk_id, {"gender": "unknown", "nationality": "unknown"})
                speaker_rows.append((spk_id, spk_meta["gender"], spk_meta["nationality"]))
            
            # Fast batch insert for speakers
            with cur.copy("COPY speakers (speaker_id, gender, nationality) FROM STDIN") as copy:
                for row in speaker_rows:
                    copy.write_row(row)
            
            print(f"Successfully inserted {len(unique_speakers)} unique speakers.")

            # --- STEP 2: Populate Audio Embeddings (Fact Table) ---
            print("Batch-inserting audio embeddings...")
            
            # Helper function to convert float list to pgvector string format: "[0.1,0.2,...]"
            def format_vector(vector_list):
                return "[" + ",".join(map(str, vector_list)) + "]"

            # Generator to stream data over to the container efficiently
            def embedding_records_generator():
                for i in range(total_records):
                    formatted_emb = format_vector(embeddings[i])
                    yield (speakers[i], processed_files[i], formatted_emb)

            with cur.copy("COPY audio_embeddings (speaker_id, file_path, embedding) FROM STDIN") as copy:
                for row in embedding_records_generator():
                    copy.write_row(row)

            print(f"Successfully inserted all {total_records} vector records!")
            
    print("Data migration pipeline completed seamlessly!")


if __name__ == "__main__":
    load_data_to_postgres()