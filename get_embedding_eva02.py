import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel
from transformers import CLIPProcessor, CLIPModel
import open_clip
import torch.nn.functional as F

# ====== 1. Load model ======
model_name = "hf-hub:timm/eva02_large_patch14_clip_224.merged2b_s4b_b131k"
model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
    model_name, pretrained="laion2b_s4b_b131k"
)
tokenizer = open_clip.get_tokenizer(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()

model.eval()

# ====== 2. Đường dẫn root folder ======
ROOT = "./keyframes"   # root chứa nhiều folder con

# ====== 3. Hàm tính embedding cho 1 folder ======
def embed_folder(folder_path, save_path, batch_size=32):
    image_files = sorted(
        [os.path.join(folder_path, f) 
         for f in os.listdir(folder_path) 
         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    )

    embeddings = []
    batch_images = []
    
    for img_path in tqdm(image_files, desc=f"Embedding {os.path.basename(folder_path)}"):
        try:
            image = Image.open(img_path).convert("RGB")
            batch_images.append(preprocess_val(image))  # preprocess ảnh

            # Nếu đủ batch thì xử lý
            if len(batch_images) == batch_size:
                batch_tensor = torch.stack(batch_images).to(device)
                with torch.no_grad():
                    feats = model.encode_image(batch_tensor)
                    feats = F.normalize(feats, dim=-1)
                embeddings.append(feats.cpu().numpy())
                batch_images = []

        except Exception as e:
            print(f"⚠️ Lỗi với ảnh {img_path}: {e}")

    # Xử lý batch cuối
    if batch_images:
        batch_tensor = torch.stack(batch_images).to(device)
        with torch.no_grad():
            feats = model.encode_image(batch_tensor)
            feats = F.normalize(feats, dim=-1)
        embeddings.append(feats.cpu().numpy())

    if embeddings:
        embeddings = np.concatenate(embeddings, axis=0)
        np.save(save_path, embeddings)
        print(f"✅ Saved {save_path}, shape={embeddings.shape}")
    else:
        print(f"❌ Folder {folder_path} không có ảnh hợp lệ!")

# ====== 4. Chạy cho tất cả folder con ======
output_dir = f"./Batch2-btc/{model_name.split('/')[-1]}"
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