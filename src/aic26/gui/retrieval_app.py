"""
Desktop GUI for EVA-02 Image Retrieval.

Provides a full-featured desktop interface for:
- Text search (single and multi-query)
- Image search
- TRAKE temporal search
- Result browsing and export

Migrated from: eva02_retrieval-ui_trake.py
"""
from __future__ import annotations

import os
import time
import customtkinter as ctk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk

from loguru import logger

from aic26.config import settings
from aic26.core.retrieval import EVA02ImageRetrieval


class EVA02GUI(ctk.CTk):
    """
    Main desktop GUI for the EVA-02 retrieval system.
    
    Features tabs for:
    - Single text query
    - Multi text query
    - Image query
    - TRAKE temporal search
    - Stored queries browser
    """
    
    def __init__(self, config=None) -> None:
        """
        Initialize the GUI application.
        
        Args:
            config: Settings instance
        """
        super().__init__()
        
        if config is None:
            config = settings
        
        self.config = config
        
        self.title("EVA-02 Image Retrieval (with TRAKE)")
        self.geometry("1280x860")
        
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Initialize retrieval engine
        logger.info("Initializing retrieval engine for GUI...")
        self.retriever = EVA02ImageRetrieval(
            embedding_dir=str(config.embedding_dir),
            keyframes_dir=str(config.keyframes_dir),
            db_path=str(config.db_path),
        )
        logger.info("Retrieval engine ready.")
        
        self._build_sidebar()
        self._build_main_area()
        
        # Thumbnail references (prevent garbage collection)
        self.tab_thumbs: dict = {}
    
    def _build_sidebar(self) -> None:
        """Build the navigation sidebar."""
        self.sidebar = ctk.CTkFrame(self, width=220, corner_radius=10)
        self.sidebar.pack(side="left", fill="y", padx=10, pady=10)
        
        ctk.CTkLabel(
            self.sidebar,
            text="EVA-02 Retrieval",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).pack(pady=16)
        
        buttons = [
            ("🔍 Single Query (Text)", self.show_single_query),
            ("📝 Multi Query (Text)", self.show_multi_query),
            ("🖼️ Image Query", self.show_image_query),
            ("⏱️ TRAKE (Temporal)", self.show_trake),
            ("📚 Stored Queries", self.show_stored_queries),
            ("📤 Export Database", self._export_database),
            ("📊 Stats", self._show_stats),
        ]
        
        for text, command in buttons:
            ctk.CTkButton(
                self.sidebar,
                text=text,
                command=command,
            ).pack(pady=6, fill="x")
        
        # Exit button at bottom
        ctk.CTkButton(
            self.sidebar,
            text="❌ Exit",
            fg_color="red",
            hover_color="darkred",
            command=self.quit,
        ).pack(side="bottom", pady=16, fill="x")
    
    def _build_main_area(self) -> None:
        """Build the main tabbed area."""
        self.main_frame = ctk.CTkFrame(self, corner_radius=10)
        self.main_frame.pack(side="left", expand=True, fill="both", padx=10, pady=10)
        
        self.tabview = ctk.CTkTabview(self.main_frame)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=10)
        self.tabview.add("Search Results")
        self.tabview.set("Search Results")
    
    def run(self) -> None:
        """Start the GUI main loop."""
        self.mainloop()
    
    # ── Tab management ────────────────────────────────────────
    
    def _clear_tab(self, tab_name: str = "Search Results") -> None:
        """Clear all widgets from a tab."""
        for w in self.tabview.tab(tab_name).winfo_children():
            w.destroy()
        self.tab_thumbs[tab_name] = []
    
    def _build_results_panel(
        self,
        results: list,
        search_time: float = None,
        allow_export: bool = True,
    ) -> None:
        """Build a scrollable results panel with thumbnails."""
        tab = "Search Results"
        self._clear_tab(tab)
        
        container = self.tabview.tab(tab)
        
        # Top bar
        top = ctk.CTkFrame(container, corner_radius=8)
        top.pack(fill="x", padx=10, pady=10)
        
        if search_time is not None:
            ctk.CTkLabel(
                top,
                text=f"Search time: {search_time:.2f}s",
                font=ctk.CTkFont(size=12),
            ).pack(side="left", padx=10)
        
        if allow_export:
            self._add_export_controls(top, results)
        
        # Scrollable results
        canvas = ctk.CTkCanvas(container, bg="#1f1f1f", highlightthickness=0)
        vsb = ctk.CTkScrollbar(container, orientation="vertical", command=canvas.yview)
        list_frame = ctk.CTkFrame(canvas)
        
        list_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=list_frame, anchor="nw")
        canvas.configure(yscrollcommand=vsb.set)
        canvas.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        
        thumbs = []
        self.tab_thumbs[tab] = thumbs
        
        for r in results:
            self._add_result_row(list_frame, r, thumbs)
        
        self.tabview.set(tab)
    
    def _add_export_controls(self, parent, results: list) -> None:
        """Add export CSV controls."""
        name_entry = ctk.CTkEntry(parent, placeholder_text="CSV name (without .csv)")
        name_entry.pack(side="left", padx=10)
        
        value_entry = ctk.CTkEntry(parent, placeholder_text="Optional const value")
        value_entry.pack(side="left", padx=10)
        
        def do_export():
            name = (name_entry.get() or "").strip()
            const_val = (value_entry.get() or "").strip()
            const_val_int = None
            
            if const_val:
                try:
                    const_val_int = int(const_val)
                except ValueError:
                    messagebox.showerror("Error", "Const value must be integer")
                    return
            
            if not name:
                messagebox.showwarning("Warning", "Please enter CSV name")
                return
            
            try:
                self.retriever.export_results(results, name, const_val_int)
                messagebox.showinfo("Success", f"Exported to ./submission/{name}.csv")
            except Exception as e:
                messagebox.showerror("Export error", str(e))
        
        ctk.CTkButton(parent, text="📄 Export CSV", command=do_export).pack(
            side="left", padx=10
        )
    
    def _add_result_row(self, parent, r: dict, thumbs: list) -> None:
        """Add a single result row with thumbnail."""
        try:
            img = Image.open(r["image_path"]).convert("RGB")
            img.thumbnail((220, 220))
            tk_img = ImageTk.PhotoImage(img)
            thumbs.append(tk_img)
            
            row = ctk.CTkFrame(parent, corner_radius=8)
            row.pack(fill="x", padx=10, pady=6)
            
            lbl = ctk.CTkLabel(row, image=tk_img, text="")
            lbl.pack(side="left", padx=8, pady=8)
            lbl.bind(
                "<Button-1>",
                lambda e, v=r["video_name"]: self._open_video_frames_tab(v),
            )
            
            sim = r.get("similarity")
            if isinstance(sim, (int, float)):
                info = (
                    f"{r['video_name']} | "
                    f"Frame {r['frame_idx']} | "
                    f"Score {sim:.3f}"
                )
            else:
                info = f"{r['video_name']} | Frame {r['frame_idx']}"
            
            ctk.CTkLabel(row, text=info, anchor="w").pack(side="left", padx=10)
            
        except Exception as e:
            logger.warning(f"Load error {r.get('image_path')}: {e}")
    
    # ── Search panels ─────────────────────────────────────────
    
    def show_single_query(self) -> None:
        """Show single text query panel."""
        self._clear_tab()
        container = self.tabview.tab("Search Results")
        
        ctk.CTkLabel(
            container,
            text="Single Query (Text)",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).pack(pady=10)
        
        entry = ctk.CTkEntry(container, placeholder_text="Enter your query…")
        entry.pack(pady=10, fill="x", padx=20)
        
        topk_entry = ctk.CTkEntry(container, placeholder_text="Top-K (default 100)")
        topk_entry.pack(pady=6, padx=20)
        
        def run():
            q = (entry.get() or "").strip()
            if not q:
                messagebox.showwarning("Warning", "Please enter a query")
                return
            
            try:
                topk = int(topk_entry.get()) if topk_entry.get().strip() else 100
            except ValueError:
                messagebox.showerror("Error", "Top-K must be integer")
                return
            
            start = time.time()
            results = self.retriever.search_text(q, top_k=topk, save_to_db=True)
            elapsed = time.time() - start
            
            if results:
                self._build_results_panel(results, search_time=elapsed)
            else:
                messagebox.showinfo("Info", "No results found")
        
        ctk.CTkButton(container, text="Search", command=run).pack(pady=10)
    
    def show_multi_query(self) -> None:
        """Show multi text query panel."""
        self._clear_tab()
        container = self.tabview.tab("Search Results")
        
        ctk.CTkLabel(
            container,
            text="Multi Query (Text)",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).pack(pady=10)
        
        txt = ctk.CTkTextbox(container, height=220)
        txt.pack(fill="both", padx=20, pady=10)
        txt.insert("1.0", "Enter one query per line…")
        
        topk_entry = ctk.CTkEntry(container, placeholder_text="Top-K per query (default 100)")
        topk_entry.pack(pady=6, padx=20)
        
        def run():
            raw = txt.get("1.0", "end").strip()
            queries = [l.strip() for l in raw.splitlines() if l.strip()]
            
            if not queries:
                messagebox.showwarning("Warning", "Please enter at least one query")
                return
            
            try:
                topk = int(topk_entry.get()) if topk_entry.get().strip() else 100
            except ValueError:
                messagebox.showerror("Error", "Top-K must be integer")
                return
            
            all_results = []
            start = time.time()
            for q in queries:
                all_results.extend(
                    self.retriever.search_text(q, top_k=topk, save_to_db=True)
                )
            elapsed = time.time() - start
            
            if all_results:
                self._build_results_panel(all_results, search_time=elapsed)
            else:
                messagebox.showinfo("Info", "No results found")
        
        ctk.CTkButton(container, text="Run Multi-Query", command=run).pack(pady=10)
    
    def show_image_query(self) -> None:
        """Show image query panel."""
        self._clear_tab()
        container = self.tabview.tab("Search Results")
        
        ctk.CTkLabel(
            container,
            text="Image Query",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).pack(pady=10)
        
        topk_entry = ctk.CTkEntry(container, placeholder_text="Top-K (default 100)")
        topk_entry.pack(pady=6, padx=20)
        
        preview = ctk.CTkLabel(container, text="No image selected")
        preview.pack(pady=10)
        
        selected = {"path": None}
        
        def pick():
            p = filedialog.askopenfilename(
                title="Choose an image",
                filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")],
            )
            if not p:
                return
            selected["path"] = p
            try:
                img = Image.open(p).convert("RGB")
                img.thumbnail((320, 320))
                imgtk = ImageTk.PhotoImage(img)
                preview.configure(image=imgtk, text="")
                preview.image = imgtk
            except Exception as e:
                preview.configure(text=f"Preview error: {e}")
        
        def run():
            if not selected["path"]:
                messagebox.showwarning("Warning", "Please select image")
                return
            
            try:
                topk = int(topk_entry.get()) if topk_entry.get().strip() else 100
            except ValueError:
                messagebox.showerror("Error", "Top-K must be integer")
                return
            
            start = time.time()
            results = self.retriever.search_image(
                selected["path"], top_k=topk, save_to_db=True,
            )
            elapsed = time.time() - start
            
            if results:
                self._build_results_panel(results, search_time=elapsed)
            else:
                messagebox.showinfo("Info", "No results found")
        
        ctk.CTkButton(container, text="Choose Image…", command=pick).pack(pady=6)
        ctk.CTkButton(container, text="Search by Image", command=run).pack(pady=6)
    
    def show_trake(self) -> None:
        """Show TRAKE temporal search panel."""
        self._clear_tab()
        container = self.tabview.tab("Search Results")
        
        ctk.CTkLabel(
            container,
            text="TRAKE (Temporal Retrieval)",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).pack(pady=10)
        
        ctk.CTkLabel(
            container,
            text="Enter events (one per line) in temporal order:",
        ).pack(pady=4)
        
        txt = ctk.CTkTextbox(container, height=260)
        txt.pack(fill="both", padx=20, pady=10)
        txt.insert("1.0", "E1: …\nE2: …\nE3: …")
        
        topk_entry = ctk.CTkEntry(container, placeholder_text="Top-K per event (default 200)")
        topk_entry.pack(pady=6, padx=20)
        
        def run():
            raw = txt.get("1.0", "end").strip()
            events = [l.strip() for l in raw.splitlines() if l.strip()]
            
            if not events:
                messagebox.showwarning("Warning", "Please enter at least one event")
                return
            
            try:
                topk = int(topk_entry.get()) if topk_entry.get().strip() else 200
            except ValueError:
                messagebox.showerror("Error", "Top-K must be integer")
                return
            
            start = time.time()
            sols = self.retriever.trake_highest(events, top_k=topk, candidates=3)
            elapsed = time.time() - start
            
            if not sols:
                messagebox.showinfo("No match", "Could not find any ordered sequence.")
                return
            
            # Display solutions
            summary = (
                f"Found {len(sols)} sequences | "
                f"Search time: {elapsed:.2f}s\n\n"
            )
            for i, sol in enumerate(sols, 1):
                summary += (
                    f"Solution {i}: {sol['video_name']} "
                    f"(Score: {sol['score']:.3f})\n"
                )
                for d in sol.get('details', []):
                    summary += f"  - {d['event'][:50]}... → Frame {d['frame']}\n"
                summary += "\n"
            
            ctk.CTkLabel(
                container,
                text=summary,
                font=ctk.CTkFont(size=11),
                justify="left",
            ).pack(pady=10, padx=20)
        
        ctk.CTkButton(container, text="Run TRAKE", command=run).pack(pady=8)
    
    def show_stored_queries(self) -> None:
        """Show stored queries panel."""
        self._clear_tab()
        container = self.tabview.tab("Search Results")
        
        ctk.CTkLabel(
            container,
            text="Stored Queries",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).pack(pady=10)
        
        queries = self.retriever.get_stored_queries()
        
        if not queries:
            ctk.CTkLabel(
                container,
                text="No stored queries",
                font=ctk.CTkFont(size=12),
            ).pack(pady=20)
            return
        
        for i, q in enumerate(queries):
            def make_cmd(idx=i):
                return lambda: self._load_query(idx)
            
            ctk.CTkButton(
                container,
                text=f"[{i}] {q[:80]}...",
                command=make_cmd(),
            ).pack(pady=4, fill="x")
    
    def _open_video_frames_tab(self, video_name: str) -> None:
        """Open a tab showing all frames of a video."""
        frames = self.retriever.get_frames_of_video(video_name)
        
        if not frames:
            messagebox.showinfo("Info", f"No frames found for {video_name}")
            return
        
        tab_name = f"Frames:{video_name}"
        if tab_name in self.tabview._tab_dict:
            try:
                self.tabview.delete(tab_name)
            except Exception:
                pass
        
        self.tabview.add(tab_name)
        self.tabview.set(tab_name)
        panel = self.tabview.tab(tab_name)
        
        canvas = ctk.CTkCanvas(panel, bg="#1f1f1f", highlightthickness=0)
        vsb = ctk.CTkScrollbar(panel, orientation="vertical", command=canvas.yview)
        list_frame = ctk.CTkFrame(canvas)
        
        list_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=list_frame, anchor="nw")
        canvas.configure(yscrollcommand=vsb.set)
        canvas.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        
        thumbs = []
        self.tab_thumbs[tab_name] = thumbs
        
        for r in frames[:50]:  # Limit to 50 frames
            try:
                img = Image.open(r["image_path"]).convert("RGB")
                img.thumbnail((180, 180))
                tk_img = ImageTk.PhotoImage(img)
                thumbs.append(tk_img)
                
                row = ctk.CTkFrame(list_frame, corner_radius=8)
                row.pack(fill="x", padx=8, pady=4)
                ctk.CTkLabel(row, image=tk_img, text="").pack(side="left", padx=6, pady=6)
                ctk.CTkLabel(
                    row,
                    text=f"{r['video_name']} | Frame {r['frame_idx']}",
                    anchor="w",
                ).pack(side="left", padx=8)
                
            except Exception as e:
                logger.warning(f"Error loading frame: {e}")
    
    def _load_query(self, idx: int) -> None:
        """Load and display results from a stored query."""
        results = self.retriever.reload_query_results(idx)
        if results:
            self.tabview.set("Search Results")
            self._build_results_panel(results, search_time=None)
        else:
            messagebox.showinfo("Info", "Invalid query index or no results")
    
    def _export_database(self) -> None:
        """Export vector database to files."""
        self.retriever.export_database("./exported_vectors")
        messagebox.showinfo("Export", "Database exported.")
    
    def _show_stats(self) -> None:
        """Show system statistics."""
        self._clear_tab()
        s = self.retriever.get_stats()
        info = (
            f"Total images: {s['total_images']}\n"
            f"Total videos: {s['total_videos']}\n"
            f"Embedding shape: {s['embedding_shape']}\n"
            f"DB queries: {s.get('database_queries', 'N/A')}"
        )
        ctk.CTkLabel(
            self.tabview.tab("Search Results"),
            text=info,
            font=ctk.CTkFont(size=13),
        ).pack(pady=20)


if __name__ == "__main__":
    app = EVA02GUI()
    app.run()