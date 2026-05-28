import os
import csv
import torch
import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector 

# Load variables from .env file
load_dotenv()

# --- DATABASE CONFIGURATION ---
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

# Dynamically build the connection string
DB_CONN_STRING = f"host={DB_HOST} port={DB_PORT} dbname={DB_NAME} user={DB_USER} password={DB_PASSWORD}"

# --- SYSTEM PATHS ---
PT_FILE_PATH = os.getenv("PT_FILE_PATH")
METADATA_CSV_PATH = os.getenv("METADATA_CSV_PATH")


def load_metadata_lookup(csv_path):
    """
    Reads the VoxCeleb metadata file and builds a high-speed
    lookup dictionary mapped by Speaker ID.
    Auto-detects if the file is comma-separated or tab-separated.
    """
    metadata_lookup = {}
    
    if not csv_path or not os.path.exists(csv_path):
        print(f"Warning: Metadata file not found at '{csv_path}'. Defaulting to 'unknown'.")
        return metadata_lookup

    with open(csv_path, mode='r', encoding='utf-8') as f:
        # Peek at the header line to see if it contains tabs or commas
        first_line = f.readline()
        delimiter = '\t' if '\t' in first_line else ','
        f.seek(0)  # Reset file pointer back to the beginning
        
        print(f"Auto-detected delimiter for metadata: {'[TAB]' if delimiter == '\t' else '[COMMA]'}")
        
        reader = csv.DictReader(f, delimiter=delimiter)
        
        for row in reader:
            spk_id = row.get('VoxCeleb1 ID', '').strip()
            gender = row.get('Gender', 'unknown').strip().lower()
            nationality = row.get('Nationality', 'unknown').strip()
            
            # Standardize gender strings to match database constraints
            gender_full = "male" if gender in ["m", "male"] else "female" if gender in ["f", "female"] else "unknown"
            
            if spk_id:
                metadata_lookup[spk_id] = {
                    "gender": gender_full,
                    "nationality": nationality
                }
                
    print(f"Successfully indexed real metadata for {len(metadata_lookup)} speakers.")
    return metadata_lookup


def load_data_to_postgres():
    # Defensive checks to ensure paths exist before running expensive loading operations
    if not PT_FILE_PATH or not os.path.exists(PT_FILE_PATH):
        print(f"Error: Embeddings file not found at: '{PT_FILE_PATH}'. Check your .env file.")
        return

    # Load metadata lookup with auto-detection
    metadata_map = load_metadata_lookup(METADATA_CSV_PATH)

    print("Loading .pt file into Python memory... (this might take a few seconds)")
    data = torch.load(PT_FILE_PATH, map_location=torch.device('cpu'))
    
    embeddings = data['embeddings'].tolist()  # Convert PyTorch tensor to Python lists
    speakers = data['speakers']
    processed_files = data['processed_files']
    
    total_records = len(speakers)
    print(f"Found {total_records} records to process.")

    print(f"Connecting to PostgreSQL database '{DB_NAME}' on {DB_HOST}:{DB_PORT}...")
    try:
        with psycopg.connect(DB_CONN_STRING) as conn:
            
            
            print("Registering native pgvector adapter...")
            register_vector(conn)
            
            with conn.cursor() as cur:
                
                # --- STEP 1: Populate Unique Speakers (Using REAL Metadata) ---
                print("Extracting and inserting unique speakers...")
                unique_speakers = set(speakers)
                
                speaker_rows = []
                for spk_id in unique_speakers:
                    spk_meta = metadata_map.get(spk_id, {"gender": "unknown", "nationality": "unknown"})
                    speaker_rows.append((spk_id, spk_meta["gender"], spk_meta["nationality"]))
                
                # Fast batch insert for speakers
                with cur.copy("COPY speakers (speaker_id, gender, nationality) FROM STDIN") as copy:
                    for row in speaker_rows:
                        copy.write_row(row)
                
                print(f"Successfully inserted {len(unique_speakers)} unique speakers.")

                # --- STEP 2: Populate Audio Embeddings (Fact Table) ---
                print("Batch-inserting audio embeddings...")
                
                # UPDATED: Generator now yields the native Python list directly
                def embedding_records_generator():
                    for i in range(total_records):
                        # No string formatting required! pgvector handles the list natively.
                        yield (speakers[i], processed_files[i], embeddings[i])

                with cur.copy("COPY audio_embeddings (speaker_id, file_path, embedding) FROM STDIN") as copy:
                    for row in embedding_records_generator():
                        copy.write_row(row)

                print(f"Successfully inserted all {total_records} vector records!")
                
        print("Data migration pipeline completed seamlessly!")

    except psycopg.OperationalError as e:
        print(f"Database Connection Error: {e}")
        print("Please double check that your PostgreSQL container is running and your .env credentials match.")


if __name__ == "__main__":
    load_data_to_postgres()