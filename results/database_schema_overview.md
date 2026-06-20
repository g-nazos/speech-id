# Database Schema Overview

Generated from the live PostgreSQL database configured in `data_base/.env`.

## Summary

- Database: voxceleb_speaker_id
- User: postgres
- Server: 172.19.0.2/32:5432
- Schema: public
- Tables: 4
- Total rows: 155081
- Database size: 495 MB

## Tables

### audio_embeddings
Rows: 153516

Columns
| Name | Type | Nullable | Default |
| --- | --- | --- | --- |
| id | integer | NO |  |
| speaker_id | character varying(20) | YES |  |
| file_path | text | NO |  |
| cluster_id | integer | YES |  |
| embedding | vector(192) | NO |  |

Constraints
| Name | Type | Definition |
| --- | --- | --- |
| audio_embeddings_cluster_id_fkey | f | FOREIGN KEY (cluster_id) REFERENCES cluster_centroids(cluster_id) ON DELETE SET NULL |
| audio_embeddings_speaker_id_fkey | f | FOREIGN KEY (speaker_id) REFERENCES speakers(speaker_id) ON DELETE CASCADE |
| audio_embeddings_pkey | p | PRIMARY KEY (id) |

Indexes
| Name | Definition |
| --- | --- |
| audio_embeddings_pkey | CREATE UNIQUE INDEX audio_embeddings_pkey ON public.audio_embeddings USING btree (id) |
| idx_audio_embeddings_cluster | CREATE INDEX idx_audio_embeddings_cluster ON public.audio_embeddings USING btree (cluster_id) |
| idx_audio_embeddings_speaker | CREATE INDEX idx_audio_embeddings_speaker ON public.audio_embeddings USING btree (speaker_id) |
| idx_global_vector_hnsw | CREATE INDEX idx_global_vector_hnsw ON public.audio_embeddings USING hnsw (embedding vector_cosine_ops) WITH (m='16', ef_construction='64') |

### cluster_centroids
Rows: 256

Columns
| Name | Type | Nullable | Default |
| --- | --- | --- | --- |
| cluster_id | integer | NO |  |
| centroid_vector | vector(192) | NO |  |

Constraints
| Name | Type | Definition |
| --- | --- | --- |
| cluster_centroids_pkey | p | PRIMARY KEY (cluster_id) |

Indexes
| Name | Definition |
| --- | --- |
| cluster_centroids_pkey | CREATE UNIQUE INDEX cluster_centroids_pkey ON public.cluster_centroids USING btree (cluster_id) |

### metadata_centroids
Rows: 58

Columns
| Name | Type | Nullable | Default |
| --- | --- | --- | --- |
| category | character varying(20) | NO |  |
| category_value | character varying(50) | NO |  |
| centroid_vector | vector(192) | NO |  |

Constraints
| Name | Type | Definition |
| --- | --- | --- |
| metadata_centroids_pkey | p | PRIMARY KEY (category, category_value) |

Indexes
| Name | Definition |
| --- | --- |
| metadata_centroids_pkey | CREATE UNIQUE INDEX metadata_centroids_pkey ON public.metadata_centroids USING btree (category, category_value) |

### speakers
Rows: 1251

Columns
| Name | Type | Nullable | Default |
| --- | --- | --- | --- |
| speaker_id | character varying(20) | NO |  |
| gender | character varying(10) | NO |  |
| nationality | character varying(20) | NO |  |

Constraints
| Name | Type | Definition |
| --- | --- | --- |
| speakers_pkey | p | PRIMARY KEY (speaker_id) |

Indexes
| Name | Definition |
| --- | --- |
| idx_speakers_gender_nat | CREATE INDEX idx_speakers_gender_nat ON public.speakers USING btree (gender, nationality) |
| speakers_pkey | CREATE UNIQUE INDEX speakers_pkey ON public.speakers USING btree (speaker_id) |
