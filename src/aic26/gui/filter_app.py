"""
Object Filter GUI for AIC-26.

Allows users to filter keyframes by detected objects.
Shows which frames contain specific combinations of objects.

Migrated from: object_filter.py
"""
from __future__ import annotations

import os
import json
from collections import Counter
from pathlib import Path

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image
import matplotlib.pyplot as plt

from loguru import logger


class ObjectFilterApp:
    """
    GUI application for filtering keyframes by detected objects.
    
    Features:
    - Scan all detection JSONs for object frequencies
    - Select objects via checkboxes
    - Search/filter object list
    - Display matching images in a grid
    - Confidence threshold control
    
    Usage:
        app = ObjectFilterApp(root, "./objects", "./keyframes")
        root.mainloop()
    """
    
    MAX_DISPLAY = 50
    FIG_SIZE = (15, 10)
    CONF_THRESHOLD = 0.5
    TOP_N_OBJECTS = 100
    
    def __init__(
        self,
        master: tk.Tk,
        objects_root: str = "./objects",
        keyframes_root: str = "./keyframes",
    ) -> None:
        """
        Initialize the object filter GUI.
        
        Args:
            master: Tkinter root window
            objects_root: Directory containing detection JSONs
            keyframes_root: Directory containing keyframe images
        """
        self.master = master
        self.master.title("Object Filter - AIC-26")
        
        self.objects_root = Path(objects_root)
        self.keyframes_root = Path(keyframes_root)
        
        # Count object frequencies
        logger.info("Scanning objects in dataset...")
        self.object_counts = self._count_objects()
        self.objects = [
            obj for obj, _ in self.object_counts.most_common(self.TOP_N_OBJECTS)
        ]
        logger.info(f"Found {len(self.objects)} unique objects (top {self.TOP_N_OBJECTS})")
        
        self.conf_threshold = tk.DoubleVar(value=self.CONF_THRESHOLD)
        self.selected_state = {obj: False for obj in self.objects}
        
        self._build_ui()
    
    def _build_ui(self) -> None:
        """Build the user interface."""
        # Search bar
        search_frame = tk.Frame(self.master)
        search_frame.pack(fill="x", pady=5)
        
        tk.Label(search_frame, text="🔍 Tìm object:").pack(side="left", padx=5)
        
        self.search_var = tk.StringVar()
        search_entry = tk.Entry(search_frame, textvariable=self.search_var)
        search_entry.pack(side="left", fill="x", expand=True, padx=5)
        search_entry.bind("<KeyRelease>", self._update_filter)
        
        # Scrollable checkbox area
        self.frame = tk.Frame(self.master)
        self.frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(self.frame)
        self.scrollbar = ttk.Scrollbar(
            self.frame, orient="vertical", command=self.canvas.yview
        )
        self.scrollable_frame = tk.Frame(self.canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        # Checkboxes
        self.vars = {}
        self.checkbuttons = {}
        self._populate_checkbuttons(self.objects)
        
        # Control panel
        control_frame = tk.Frame(self.master)
        control_frame.pack(pady=10)
        
        tk.Label(control_frame, text="Confidence threshold:").pack(side="left")
        tk.Entry(control_frame, textvariable=self.conf_threshold, width=5).pack(
            side="left", padx=5
        )
        
        tk.Button(
            control_frame, text="🔍 Tìm ảnh", command=self._search_images
        ).pack(side="left", padx=5)
        
        tk.Button(
            control_frame, text="❌ Thoát", command=self.master.quit
        ).pack(side="left", padx=5)
    
    def _count_objects(self) -> Counter:
        """Count object frequencies across all detection JSONs."""
        counter = Counter()
        
        if not self.objects_root.exists():
            logger.warning(f"Objects directory not found: {self.objects_root}")
            return counter
        
        for video_folder in os.listdir(str(self.objects_root)):
            json_dir = self.objects_root / video_folder
            if not json_dir.is_dir():
                continue
            
            for file in os.listdir(str(json_dir)):
                if not file.endswith(".json"):
                    continue
                
                json_path = json_dir / file
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        detections = json.load(f)
                    entities = detections.get("detection_class_entities", [])
                    counter.update(entities)
                except Exception as e:
                    logger.warning(f"Error reading {json_path}: {e}")
        
        return counter
    
    def _populate_checkbuttons(self, objects: list) -> None:
        """Create checkboxes for object list."""
        for cb in self.checkbuttons.values():
            cb.destroy()
        self.vars.clear()
        self.checkbuttons.clear()
        
        for obj in objects:
            var = tk.BooleanVar(value=self.selected_state.get(obj, False))
            chk = tk.Checkbutton(
                self.scrollable_frame,
                text=obj,
                variable=var,
                command=lambda o=obj, v=var: self._update_state(o, v),
            )
            chk.pack(anchor="w")
            self.vars[obj] = var
            self.checkbuttons[obj] = chk
    
    def _update_state(self, obj: str, var: tk.BooleanVar) -> None:
        """Update selected state for an object."""
        self.selected_state[obj] = var.get()
    
    def _update_filter(self, event=None) -> None:
        """Filter checkboxes by search query."""
        query = self.search_var.get().lower()
        if not query:
            filtered = self.objects
        else:
            filtered = [obj for obj in self.objects if query in obj.lower()]
        self._populate_checkbuttons(filtered)
    
    def _search_images(self) -> None:
        """Search for images containing all selected objects."""
        selected = [obj for obj, state in self.selected_state.items() if state]
        
        if not selected:
            messagebox.showwarning("Chưa chọn", "Bạn chưa chọn object nào!")
            return
        
        threshold = float(self.conf_threshold.get())
        logger.info(
            f"Searching images with ALL objects {selected} (conf ≥ {threshold})"
        )
        
        matched_images = []
        
        for video_folder in os.listdir(str(self.objects_root)):
            json_dir = self.objects_root / video_folder
            img_dir = self.keyframes_root / video_folder
            
            if not json_dir.is_dir() or not img_dir.is_dir():
                continue
            
            for file in os.listdir(str(json_dir)):
                if not file.endswith(".json"):
                    continue
                
                json_path = json_dir / file
                with open(json_path, "r", encoding="utf-8") as f:
                    detections = json.load(f)
                
                entities = detections.get("detection_class_entities", [])
                scores = detections.get("detection_scores", [])
                
                present_objs = {
                    obj for obj, score in zip(entities, scores)
                    if float(score) >= threshold
                }
                
                if all(obj in present_objs for obj in selected):
                    base = Path(file).stem
                    for ext in [".jpg", ".png", ".jpeg"]:
                        img_path = img_dir / (base + ext)
                        if img_path.exists():
                            matched_images.append(str(img_path))
                            break
        
        messagebox.showinfo("Kết quả", f"Tìm thấy {len(matched_images)} ảnh phù hợp.")
        
        if matched_images:
            self._display_results(matched_images)
    
    def _display_results(self, image_paths: list) -> None:
        """Display matching images in a matplotlib grid."""
        max_display = min(len(image_paths), self.MAX_DISPLAY)
        
        rows = cols = int(max_display ** 0.5) + 1
        fig, axes = plt.subplots(rows, cols, figsize=self.FIG_SIZE)
        axes = axes.flatten()
        
        for i in range(max_display):
            try:
                img = Image.open(image_paths[i]).convert("RGB")
                axes[i].imshow(img)
                axes[i].axis("off")
                axes[i].set_title(
                    os.path.basename(image_paths[i]), fontsize=8
                )
            except Exception as e:
                logger.warning(f"Display error: {e}")
        
        for j in range(max_display, len(axes)):
            axes[j].axis("off")
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectFilterApp(root)
    root.mainloop()