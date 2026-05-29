import os
import csv
import struct
import torch
import psycopg
from dotenv import load_dotenv

load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

DB_CONN_STRING = f"host={DB_HOST} port={DB_PORT} dbname={DB_NAME} user={DB_USER} password={DB_PASSWORD}"

PT_FILE_PATH = os.getenv("PT_FILE_PATH")
METADATA_CSV_PATH = os.getenv("METADATA_CSV_PATH")


def load_metadata_lookup(csv_path):
    """Parses VoxCeleb metadata into a dictionary mapped by Speaker ID."""
    metadata_lookup = {}
    
    if not csv_path or not os.path.exists(csv_path):
        print(f"Warning: Metadata file not found at '{csv_path}'. Defaulting to 'unknown'.")
        return metadata_lookup

    with open(csv_path, mode='r', encoding='utf-8') as f:
        first_line = f.readline()
        delimiter = '\t' if '\t' in first_line else ','
        f.seek(0)
        
        reader = csv.DictReader(f, delimiter=delimiter)
        
        for row in reader:
            spk_id = row.get('VoxCeleb1 ID', '').strip()
            gender = row.get('Gender', 'unknown').strip().lower()
            nationality = row.get('Nationality', 'unknown').strip()
            
            gender_full = "male" if gender in ["m", "male"] else "female" if gender in ["f", "female"] else "unknown"
            
            if spk_id:
                metadata_lookup[spk_id] = {
                    "gender": gender_full,
                    "nationality": nationality
                }
                
    print(f"Indexed metadata for {len(metadata_lookup)} speakers.")
    return metadata_lookup


def compute_and_save_centroids(cur):
    """Computes a two-layer hierarchy using existing metadata_centroids columns."""
    print("Generating hierarchical tree centroids...")
    
    cur.execute("TRUNCATE TABLE metadata_centroids;")
    
    # Layer 1: Global Gender Centroids
    gender_query = """
        INSERT INTO metadata_centroids (category, category_value, centroid_vector)
        SELECT 
            'gender' as category,
            s.gender as category_value,
            avg(ae.embedding)::vector(192) as centroid_vector
        FROM audio_embeddings ae
        JOIN speakers s ON ae.speaker_id = s.speaker_id
        WHERE s.gender IS NOT NULL AND s.gender != 'unknown'
        GROUP BY s.gender;
    """
    cur.execute(gender_query)
    
    # Layer 2: Dependent Gender-Nationality Centroids (e.g., 'male:US')
    gender_nat_query = """
        INSERT INTO metadata_centroids (category, category_value, centroid_vector)
        SELECT 
            'gender_nationality' as category,
            CONCAT(s.gender, ':', s.nationality) as category_value,
            avg(ae.embedding)::vector(192) as centroid_vector
        FROM audio_embeddings ae
        JOIN speakers s ON ae.speaker_id = s.speaker_id
        WHERE s.gender != 'unknown' AND s.nationality != 'unknown'
        GROUP BY s.gender, s.nationality;
    """
    cur.execute(gender_nat_query)
    print("Centroid tree generation completed.")


def load_data_to_postgres():
    if not PT_FILE_PATH or not os.path.exists(PT_FILE_PATH):
        print(f"Error: Embeddings file not found at: '{PT_FILE_PATH}'.")
        return

    metadata_map = load_metadata_lookup(METADATA_CSV_PATH)

    print("Loading tensor data into memory...")
    data = torch.load(PT_FILE_PATH, map_location=torch.device('cpu'))
    
    embeddings = data['embeddings'].tolist()
    speakers = data['speakers']
    processed_files = data['processed_files']
    
    total_records = len(speakers)
    print(f"Processing {total_records} records...")

    try:
        with psycopg.connect(DB_CONN_STRING) as conn:
            with conn.cursor() as cur:
                
                unique_speakers = set(speakers)
                speaker_rows = []
                
                for spk_id in unique_speakers:
                    spk_meta = metadata_map.get(spk_id, {"gender": "unknown", "nationality": "unknown"})
                    speaker_rows.append((spk_id, spk_meta["gender"], spk_meta["nationality"]))
                
                print("Inserting speakers...")
                cur.executemany("""
                    INSERT INTO speakers (speaker_id, gender, nationality)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (speaker_id) DO NOTHING;
                """, speaker_rows)

                print("Executing binary COPY for audio embeddings...")
                copy_query = "COPY audio_embeddings (speaker_id, file_path, embedding) FROM STDIN WITH (FORMAT BINARY)"
                
                with cur.copy(copy_query) as copy:
                    for i in range(total_records):
                        spk_id = speakers[i]
                        file_path = processed_files[i]
                        vector = embeddings[i]
                        
                        spk_bytes = spk_id.encode('utf-8')
                        path_bytes = file_path.encode('utf-8')
                        
                        dim = len(vector)
                        unused = 0
                        vector_binary_data = struct.pack(f">HH{dim}f", dim, unused, *vector)

                        copy.write_row([spk_bytes, path_bytes, vector_binary_data])
                
                compute_and_save_centroids(cur)
                conn.commit()
                
        print("Database population pipeline finished successfully.")

    except psycopg.OperationalError as e:
        print(f"Database Connection Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    load_data_to_postgres()