import os
import json
import tkinter as tk
from tkinter import ttk, messagebox
from collections import Counter
from PIL import Image
import matplotlib.pyplot as plt

# ===============================
# 🔧 CẤU HÌNH TOÀN CỤC
# ===============================

OBJECTS_ROOT = "./objects"      # thư mục gốc chứa tất cả JSON
KEYFRAMES_ROOT = "./keyframes"  # thư mục gốc chứa tất cả ảnh
MAX_DISPLAY = 50                # giới hạn số ảnh hiển thị
FIG_SIZE = (15, 10)             # kích thước grid hiển thị
CONF_THRESHOLD = 0.5            # ngưỡng confidence mặc định
TOP_N_OBJECTS = 100             # chỉ lấy top N object phổ biến nhất


# ===============================
# APP LỌC OBJECT
# ===============================
class ObjectFilterApp:
    def __init__(self, master, objects_root, keyframes_root):
        self.master = master
        self.master.title("Object Filter")

        self.objects_root = objects_root
        self.keyframes_root = keyframes_root

        # Đếm tần suất object trong toàn bộ JSON
        print("🔄 Đang quét object trong toàn bộ dataset...")
        self.object_counts = self._count_objects()
        self.objects = [obj for obj, _ in self.object_counts.most_common(TOP_N_OBJECTS)]

        self.conf_threshold = tk.DoubleVar(value=CONF_THRESHOLD)

        # Ghi nhớ tick
        self.selected_state = {obj: False for obj in self.objects}

        # Ô tìm kiếm
        search_frame = tk.Frame(self.master)
        search_frame.pack(fill="x", pady=5)
        tk.Label(search_frame, text="🔍 Tìm object:").pack(side="left", padx=5)
        self.search_var = tk.StringVar()
        search_entry = tk.Entry(search_frame, textvariable=self.search_var)
        search_entry.pack(side="left", fill="x", expand=True, padx=5)
        search_entry.bind("<KeyRelease>", self.update_filter)

        # Scrollable frame cho checkbox
        self.frame = tk.Frame(self.master)
        self.frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.frame)
        self.scrollbar = ttk.Scrollbar(self.frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Danh sách checkbox
        self.vars = {}
        self.checkbuttons = {}
        self.populate_checkbuttons(self.objects)

        # Buttons + Threshold
        control_frame = tk.Frame(self.master)
        control_frame.pack(pady=10)

        tk.Label(control_frame, text="Confidence threshold:").pack(side="left")
        tk.Entry(control_frame, textvariable=self.conf_threshold, width=5).pack(side="left", padx=5)

        tk.Button(control_frame, text="🔍 Tìm ảnh", command=self.search_images).pack(side="left", padx=5)
        tk.Button(control_frame, text="❌ Thoát", command=self.master.quit).pack(side="left", padx=5)

    def _count_objects(self):
        """Đếm số lần xuất hiện của từng object trong toàn bộ dataset"""
        counter = Counter()
        for video_folder in os.listdir(self.objects_root):
            json_dir = os.path.join(self.objects_root, video_folder)
            if not os.path.isdir(json_dir):
                continue
            for file in os.listdir(json_dir):
                if not file.endswith(".json"):
                    continue
                json_path = os.path.join(json_dir, file)
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        detections = json.load(f)
                    entities = detections.get("detection_class_entities", [])
                    counter.update(entities)
                except Exception as e:
                    print(f"⚠️ Lỗi đọc {json_path}: {e}")
        return counter

    def populate_checkbuttons(self, objects):
        """Tạo danh sách checkbox từ list objects, giữ lại tick cũ"""
        for cb in self.checkbuttons.values():
            cb.destroy()
        self.vars.clear()
        self.checkbuttons.clear()

        for obj in objects:
            var = tk.BooleanVar(value=self.selected_state.get(obj, False))
            chk = tk.Checkbutton(self.scrollable_frame, text=obj, variable=var,
                                 command=lambda o=obj, v=var: self.update_state(o, v))
            chk.pack(anchor="w")
            self.vars[obj] = var
            self.checkbuttons[obj] = chk

    def update_state(self, obj, var):
        """Cập nhật trạng thái tick"""
        self.selected_state[obj] = var.get()

    def update_filter(self, event=None):
        """Lọc checkbox theo từ khóa tìm kiếm"""
        query = self.search_var.get().lower()
        if not query:
            filtered = self.objects
        else:
            filtered = [obj for obj in self.objects if query in obj.lower()]
        self.populate_checkbuttons(filtered)

    def search_images(self):
            # Lấy object đã tick
            selected_objects = [obj for obj, state in self.selected_state.items() if state]
            if not selected_objects:
                messagebox.showwarning("Chưa chọn", "Bạn chưa chọn object nào!")
                return

            threshold = float(self.conf_threshold.get())
            print(f"🔍 Đang tìm các ảnh chứa TẤT CẢ object {selected_objects} với score ≥ {threshold}")

            matched_images = []

            # Duyệt tất cả folder con trong OBJECTS_ROOT
            for video_folder in os.listdir(self.objects_root):
                json_dir = os.path.join(self.objects_root, video_folder)
                img_dir = os.path.join(self.keyframes_root, video_folder)

                if not os.path.isdir(json_dir) or not os.path.isdir(img_dir):
                    continue

                for file in os.listdir(json_dir):
                    if not file.endswith(".json"):
                        continue
                    json_path = os.path.join(json_dir, file)
                    with open(json_path, "r", encoding="utf-8") as f:
                        detections = json.load(f)

                    entities = detections.get("detection_class_entities", [])
                    scores = detections.get("detection_scores", [])

                    # Lấy danh sách object đạt ngưỡng
                    present_objs = {obj for obj, score in zip(entities, scores)
                                    if float(score) >= threshold}

                    # Ảnh chỉ được chọn nếu chứa tất cả object đã tick
                    if all(obj in present_objs for obj in selected_objects):
                        base = os.path.splitext(file)[0]
                        for ext in [".jpg", ".png", ".jpeg"]:
                            img_path = os.path.join(img_dir, base + ext)
                            if os.path.exists(img_path):
                                matched_images.append(img_path)
                                break

            # Thông báo số lượng
            messagebox.showinfo("Kết quả", f"Tìm thấy {len(matched_images)} ảnh phù hợp.")

            if matched_images:
                self.display_results(matched_images)
    def display_results(self, image_paths):
        """Hiển thị kết quả ảnh"""
        max_display = min(len(image_paths), MAX_DISPLAY)

        rows = cols = int(max_display ** 0.5) + 1
        fig, axes = plt.subplots(rows, cols, figsize=FIG_SIZE)
        axes = axes.flatten()

        for i in range(max_display):
            img = Image.open(image_paths[i]).convert("RGB")
            axes[i].imshow(img)
            axes[i].axis("off")
            axes[i].set_title(image_paths[i], fontsize=8)

        for j in range(max_display, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        plt.show()


# ===============================
# CHẠY DEMO
# ===============================
if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectFilterApp(
        root,
        objects_root=OBJECTS_ROOT,
        keyframes_root=KEYFRAMES_ROOT
    )
    root.mainloop()
