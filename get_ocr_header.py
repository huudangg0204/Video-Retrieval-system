import os, sys
import glob

p = os.path.join(sys.prefix, "Lib", "site-packages", "paddle", "libs")
print("paddle\\libs =", p)
print("cudnn  :", [os.path.basename(x) for x in glob.glob(os.path.join(p, "cudnn*.dll"))])
print("cublas :", [os.path.basename(x) for x in glob.glob(os.path.join(p, "cublas*.dll"))])
PADDLE_LIBS = os.path.join(sys.prefix, "Lib", "site-packages", "paddle", "libs")
if os.path.isdir(PADDLE_LIBS):
    # Python 3.8+ on Windows: ensure DLL search path
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(PADDLE_LIBS)
    # also prepend to PATH for safety
    os.environ["PATH"] = PADDLE_LIBS + ";" + os.environ.get("PATH", "")
else:
    raise RuntimeError(f"Không tìm thấy paddle\\libs: {PADDLE_LIBS}")

# (tuỳ chọn) kiểm tra DLL hiện diện
for dll in ["cudnn64_8.dll", "cublas64_11.dll"]:
    dll_path = os.path.join(PADDLE_LIBS, dll)
    if not os.path.exists(dll_path):
        print(f"[CẢNH BÁO] Thiếu {dll}: {dll_path} (hãy kiểm tra lại cài đặt paddlepaddle-gpu)")
import paddle
print("Paddle CUDA:", paddle.is_compiled_with_cuda())
paddle.device.set_device('gpu:0')   # ép dùng GPU

from paddleocr import PaddleOCR
import cv2, numpy as np, json, glob, os, pandas as pd
from tqdm import tqdm
from collections import defaultdict

# --- Khởi tạo OCR 1 lần ---
ocr = PaddleOCR(
    lang='vi',
    use_angle_cls=True,          # 2.7 dùng tham số này
    det_limit_side_len=1280,     # có thể chỉnh nhỏ hơn để nhanh hơn
    rec_batch_num=64,         # tăng batch để tận dụng GPU
    #tắt chế độ ghi log
    show_log=False,
)

def detect_all_text(img_path_or_bgr, min_conf=0.30):
    """Trả về: [{'text', 'conf', 'box'(4 điểm)} ...]"""
    bgr = cv2.imread(img_path_or_bgr) if isinstance(img_path_or_bgr, str) else img_path_or_bgr
    if bgr is None:
        raise FileNotFoundError(str(img_path_or_bgr))

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    res = ocr.ocr(rgb, cls=True)

    # Nếu None hoặc list rỗng
    if res is None or not len(res):
        return []

    # Nếu res là list 2 chiều (thường gặp ở PaddleOCR 2.x)
    if isinstance(res[0], list):
        res = res[0]

    items = []
    for r in res:
        if r is None or len(r) < 2:
            continue
        box, (txt, conf) = r
        c = float(conf)
        if c < min_conf:
            continue
        items.append({
            "text": txt,
            "conf": c,
            "box": [[float(x), float(y)] for (x,y) in box]
        })

    # Sắp xếp theo dòng cho dễ đọc
    items.sort(key=lambda it: (round(min(p[1] for p in it["box"])/10), 
                               min(p[0] for p in it["box"])))
    return items


def draw_text_boxes(img_path_or_bgr, items, out_path, show_text=True):
    bgr = cv2.imread(img_path_or_bgr) if isinstance(img_path_or_bgr, str) else img_path_or_bgr
    canvas = bgr.copy()
    for it in items:
        pts = np.array(it["box"], dtype=np.int32)
        cv2.polylines(canvas, [pts], True, (0,255,0), 2, cv2.LINE_AA)
        if show_text:
            txt = f'{it["text"]} ({it["conf"]:.2f})'
            x = int(min(p[0] for p in it["box"]))
            y = int(min(p[1] for p in it["box"])) - 6
            cv2.putText(canvas, txt[:60], (x, max(y, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, cv2.LINE_AA)
    cv2.imwrite(out_path, canvas)
    return out_path

def scan_folder_all_text(root_dir, out_csv="all_text_ocr.csv", annotate=False, ann_dir="ann_all_text",
                         min_conf=0.30, limit=None):
    if annotate:
        os.makedirs(ann_dir, exist_ok=True)
    # checkpoint per-subfolder
    checkpoint_dir = os.path.join(root_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    rows = []
    folders = sorted(glob.glob(os.path.join(root_dir, "*")))
    print(f"✓ Tìm thư mục con: {len(folders)}")
    for folder in tqdm(folders, desc="Thư mục con"):
        checkpoint_path = os.path.join(checkpoint_dir, f"{os.path.basename(folder)}_{out_csv}")

        if os.path.exists(checkpoint_path):
            print(f"[BỎ QUA] Đã có checkpoint: {checkpoint_path}")
            checkpoint_df = pd.read_csv(checkpoint_path)
            rows.extend(checkpoint_df.to_dict(orient="records"))    
            continue
        files = sorted(glob.glob(os.path.join(folder, "*.jpg")) + glob.glob(os.path.join(folder, "*.png")))
        folder_name = os.path.basename(folder)
        folder_rows = defaultdict(list)
        for f in tqdm(files, desc=f"Ảnh trong {folder_name}", leave=False):
            if os.path.getsize(f) < 1024:
                print(f"[BỎ QUA] Ảnh quá nhỏ: {f}")
                continue
            items = detect_all_text(f, min_conf=min_conf)
            row = {
                "video": folder_name,
                "file": os.path.basename(p),
                "path": p,
                "text_full": " ".join(it["text"] for it in items),
                "num_boxes": len(items),
                "conf_mean": float(np.mean([it["conf"] for it in items])) if items else None,
                "boxes": json.dumps([it["box"] for it in items], ensure_ascii=False),
                "texts": json.dumps([it["text"] for it in items], ensure_ascii=False),
                "confs": json.dumps([it["conf"] for it in items], ensure_ascii=False),
            }
            rows.append(row)
            folder_rows[folder_name].append(row)

            # Annotate into per-folder subdir if requested
            if annotate:
                sub_ann = os.path.join(ann_dir, folder_name)
                os.makedirs(sub_ann, exist_ok=True)
                out_img = os.path.join(sub_ann, os.path.basename(f))
                draw_text_boxes(f, items, out_img, show_text=False)

        # Write checkpoint CSV for this folder (so we can resume / inspect progress)
        try:
            pd.DataFrame(folder_rows[folder_name]).to_csv(checkpoint_path, index=False, encoding="utf-8")
        except Exception as e:
            # non-fatal: print a warning and continue
            print(f"[WARN] Không thể lưu checkpoint {checkpoint_path}: {e}")

    # Full CSV
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"✓ Lưu CSV: {os.path.abspath(out_csv)} | Ảnh: {len(files)}")
    if annotate:
        print(f"✓ Ảnh gán bbox: {os.path.abspath(ann_dir)}")
    print(f"✓ Checkpoints per-folder: {os.path.abspath(checkpoint_dir)}")
    return df

# # 1) Một ảnh
# img_path = "./keyframes/L30_V079/016.jpg"   # có thể sửa
# items = detect_all_text(img_path, min_conf=0.35)
# print(len(items), "boxes")
# for it in items[:5]:
#     print(it["text"], it["conf"])

# # Lưu ảnh có vẽ bbox:
# draw_text_boxes(img_path, items, out_path="/kaggle/working/0017_annot.jpg")

# 2) Cả thư mục
ROOT = "./keyframes"      # thư mục chứa ảnh (có thể sửa)
df = scan_folder_all_text(ROOT, out_csv="all_text_ocr.csv",
                          annotate=True, ann_dir="ann_all_text",
                          min_conf=0.35)
print(df.head())