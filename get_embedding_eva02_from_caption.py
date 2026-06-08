import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel
from transformers import CLIPProcessor, CLIPModel
import open_clip
import torch.nn.functional as F
import json
# ====== 1. Load model ======
model_name = "hf-hub:timm/eva02_large_patch14_clip_224.merged2b_s4b_b131k"
model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
    model_name, pretrained="laion2b_s6b_b61k"
)
tokenizer = open_clip.get_tokenizer(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()

model.eval()

# ====== 2. Đường dẫn root folder ======
ROOT = "./blip_index/checkpoints"   # root chứa nhiều folder con

# ====== 3. Hàm tính embedding cho 1 folder ======
def embed_folder(folder_path, save_path, batch_size=32):
    # load file json trong folder
    json_files = sorted(
        [os.path.join(folder_path, f) 
         for f in os.listdir(folder_path) 
         if f.lower().endswith('.json')]
    )
    if not json_files:
        print(f"❌ No json file in {folder_path}")
        return
    json_file = json_files[0]

    embeddings = []
    batch_captions = []

    # Robust JSON loading: try normal, then JSON lines or concatenated objects
    with open(json_file, 'r', encoding='utf-8') as f:
        text = f.read()

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            data = [data]
    except json.JSONDecodeError:
        # try JSON Lines (one JSON object per line)
        data = []
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for line in lines:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                # last resort: try to convert concatenated objects into an array
                try:
                    data = json.loads("[" + text.replace("}\n{", "},{").replace("}{", "},{") + "]")
                    break
                except json.JSONDecodeError as e:
                    raise ValueError(f"Failed to parse JSON file {json_file}: {e}")

    for item in tqdm(data, desc=f"Embedding {os.path.basename(folder_path)}"):
        try:
            caption = item.get("caption", "")
            if not isinstance(caption, str):
                caption = str(caption)
            batch_captions.append(caption)

            # Nếu đủ batch thì xử lý
            if len(batch_captions) == batch_size:
                tokens = open_clip.tokenize(batch_captions).to(device)
                with torch.no_grad():
                    feats = model.encode_text(tokens)
                    feats = F.normalize(feats, dim=-1)
                embeddings.append(feats.cpu().numpy())
                batch_captions = []

        except Exception as e:
            print(f"⚠️ Lỗi với json {item}: {e}")

    # Xử lý batch cuối
    if batch_captions:
        tokens = open_clip.tokenize(batch_captions).to(device)
        with torch.no_grad():
            feats = model.encode_text(tokens)
            feats = F.normalize(feats, dim=-1)
        embeddings.append(feats.cpu().numpy())

    if embeddings:
        embeddings = np.concatenate(embeddings, axis=0)
        np.save(save_path, embeddings)
        print(f"✅ Saved {save_path}, shape={embeddings.shape}")
    else:
        print(f"❌ Folder {folder_path} không có caption hợp lệ!")

# ====== 4. Chạy cho tất cả folder con ======
output_dir = f"./caption_{model_name.split('/')[-1]}"
os.makedirs(output_dir, exist_ok=True)

folders = [f for f in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, f))]

for folder in tqdm(folders, desc="Folders", unit="folder"):
    folder_path = os.path.join(ROOT, folder)
    save_file = os.path.join(output_dir, f"{folder}.npy")

    if os.path.exists(save_file):
        tqdm.write(f"Skipping (exists): {folder}")
        continue

    try:
        embed_folder(folder_path, save_file, batch_size=40)  # chỉnh batch_size tuỳ GPU
        tqdm.write(f"✅ Finished processing folder: {folder}")
    except Exception as e:
        tqdm.write(f"⚠️ Error processing {folder}: {e}")