import os
import glob
import faiss
import numpy as np
import pickle

DB_PATH = "faiss_db.pkl"

def load_embeddings_from_folder(folder_path: str) -> np.ndarray:
    npy_files = glob.glob(os.path.join(folder_path, "*.npy"))
    all_embeddings = []
    for file in npy_files:
        emb = np.load(file)
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        all_embeddings.append(emb)
    if not all_embeddings:
        raise ValueError(f"No .npy files found in folder: {folder_path}")
    return np.vstack(all_embeddings)

def create_faiss_database(embeddings: np.ndarray):
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings.astype(np.float32))
    db_data = {
        'queries': [],
        'embeddings': embeddings.astype(np.float32),
        'image_paths': [],
        'video_names': [],
        'frame_indices': [],
        'similarities': [],
        'timestamps': [],
        'metadata': {},
        'faiss_index': index
    }
    with open(DB_PATH, "wb") as f:
        pickle.dump(db_data, f)

if __name__ == "__main__":
    folder_path = r"./Batch2-btc/eva02_large_patch14_clip_224.merged2b_s4b_b131k"
    embeddings = load_embeddings_from_folder(folder_path)
    print(f"Total embeddings loaded: {embeddings.shape[0]}, dim: {embeddings.shape[1]}")
    create_faiss_database(embeddings)
    print("Index saved to faiss_db.pkl")
