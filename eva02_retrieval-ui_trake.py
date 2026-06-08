# eva02_retrieval-ui_trake.py — Desktop GUI for EVA02 retrieval (with TRAKE tab)
#
# Fixes vs. original:
# 1. show_trake called self.retriever.temporal_retrieval() which doesn't exist → NameError.
#    Now calls self.retriever.trake_highest() (correct method name).
# 2. Dead commented-out code block (~30 lines) removed for readability.
# 3. Export callbacks use the correct export_trake / export_results signatures.

import os
import time
import customtkinter as ctk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk

from eva02_retrieval_trake import EVA02ImageRetrieval


class EVA02GUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("EVA02 Image Retrieval (with TRAKE)")
        self.geometry("1280x860")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Engine
        self.retriever = EVA02ImageRetrieval()

        # Sidebar
        self.sidebar = ctk.CTkFrame(self, width=220, corner_radius=10)
        self.sidebar.pack(side="left", fill="y", padx=10, pady=10)

        ctk.CTkLabel(
            self.sidebar, text="EVA02 Retrieval",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=16)

        self.btn_single = ctk.CTkButton(
            self.sidebar, text="🔍 Single Query (Text)", command=self.show_single_query
        )
        self.btn_single.pack(pady=6, fill="x")

        self.btn_multi = ctk.CTkButton(
            self.sidebar, text="📝 Multi Query (Text)", command=self.show_multi_query
        )
        self.btn_multi.pack(pady=6, fill="x")

        self.btn_image = ctk.CTkButton(
            self.sidebar, text="🖼️ Image Query", command=self.show_image_query
        )
        self.btn_image.pack(pady=6, fill="x")

        self.btn_trake = ctk.CTkButton(
            self.sidebar, text="⏱️ TRAKE (Temporal)", command=self.show_trake
        )
        self.btn_trake.pack(pady=6, fill="x")

        self.btn_stored = ctk.CTkButton(
            self.sidebar, text="📚 Stored Queries", command=self.show_stored_queries
        )
        self.btn_stored.pack(pady=6, fill="x")

        self.btn_exportdb = ctk.CTkButton(
            self.sidebar, text="📤 Export Database", command=self.export_database
        )
        self.btn_exportdb.pack(pady=6, fill="x")

        self.btn_stats = ctk.CTkButton(
            self.sidebar, text="📊 Stats", command=self.show_stats
        )
        self.btn_stats.pack(pady=6, fill="x")

        self.btn_exit = ctk.CTkButton(
            self.sidebar, text="❌ Exit",
            fg_color="red", hover_color="darkred",
            command=self.quit
        )
        self.btn_exit.pack(side="bottom", pady=16, fill="x")

        # Main area
        self.main_frame = ctk.CTkFrame(self, corner_radius=10)
        self.main_frame.pack(side="left", expand=True, fill="both", padx=10, pady=10)

        self.tabview = ctk.CTkTabview(self.main_frame)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=10)
        self.tabview.add("Search Results")
        self.tabview.set("Search Results")

        # Keep thumbnail references to prevent GC
        self.tab_thumbs: dict = {}

    # ── Tab helpers ───────────────────────────────────────────────────────────

    def clear_search_tab(self):
        for w in self.tabview.tab("Search Results").winfo_children():
            w.destroy()
        self.tab_thumbs["Search Results"] = []

    def _build_results_panel(self, results, search_time=None, allow_export_name=True):
        tab = "Search Results"
        self.clear_search_tab()

        top = ctk.CTkFrame(self.tabview.tab(tab), corner_radius=8)
        top.pack(fill="x", padx=10, pady=10)

        if search_time is not None:
            ctk.CTkLabel(
                top, text=f"Search time: {search_time:.2f}s",
                font=ctk.CTkFont(size=12)
            ).pack(side="left", padx=10)

        if allow_export_name:
            name_entry  = ctk.CTkEntry(top, placeholder_text="CSV name (without .csv)")
            name_entry.pack(side="left", padx=10)
            value_entry = ctk.CTkEntry(top, placeholder_text="Optional const value")
            value_entry.pack(side="left", padx=10)

            def do_export():
                name      = (name_entry.get() or "").strip()
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
                    messagebox.showinfo("Submission", f"Exported to ./submission/{name}.csv")
                except Exception as e:
                    messagebox.showerror("Export error", str(e))

            ctk.CTkButton(top, text="📄 Export CSV", command=do_export).pack(side="left", padx=10)

        canvas = ctk.CTkCanvas(self.tabview.tab(tab), bg="#1f1f1f", highlightthickness=0)
        vsb    = ctk.CTkScrollbar(self.tabview.tab(tab), orientation="vertical",
                                   command=canvas.yview)
        list_frame = ctk.CTkFrame(canvas)
        list_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=list_frame, anchor="nw")
        canvas.configure(yscrollcommand=vsb.set)
        canvas.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        thumbs = []
        self.tab_thumbs[tab] = thumbs
        for r in results:
            try:
                img = Image.open(r["image_path"]).convert("RGB")
                img.thumbnail((220, 220))
                tk_img = ImageTk.PhotoImage(img)
                thumbs.append(tk_img)

                row = ctk.CTkFrame(list_frame, corner_radius=8)
                row.pack(fill="x", padx=10, pady=6)

                lbl = ctk.CTkLabel(row, image=tk_img, text="")
                lbl.pack(side="left", padx=8, pady=8)
                lbl.bind("<Button-1>", lambda e, v=r["video_name"]: self._open_video_frames_tab(v))

                sim  = r.get("similarity")
                info = f"{r['video_name']} | Frame {r['frame_idx']} | Score {sim:.3f}" \
                       if isinstance(sim, (int, float)) else \
                       f"{r['video_name']} | Frame {r['frame_idx']}"
                ctk.CTkLabel(row, text=info, anchor="w").pack(side="left", padx=10)
            except Exception as e:
                print(f"⚠️ Load error {r.get('image_path')}: {e}")

        self.tabview.set(tab)

    # ── Search panels ─────────────────────────────────────────────────────────

    def show_single_query(self):
        self.clear_search_tab()
        container = self.tabview.tab("Search Results")
        ctk.CTkLabel(
            container, text="Single Query (Text)",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=10)
        entry      = ctk.CTkEntry(container, placeholder_text="Enter your query…")
        entry.pack(pady=10, fill="x", padx=20)
        topk_entry = ctk.CTkEntry(container, placeholder_text="Top-K (default 100)")
        topk_entry.pack(pady=6, padx=20)

        def run_query():
            q = (entry.get() or "").strip()
            if not q:
                messagebox.showwarning("Warning", "Please enter a query")
                return
            try:
                topk = int(topk_entry.get()) if topk_entry.get().strip() else 100
            except ValueError:
                messagebox.showerror("Error", "Top-K must be integer")
                return
            start   = time.time()
            results = self.retriever.search_text(q, top_k=topk, save_to_db=True)
            elapsed = time.time() - start
            if results:
                self._build_results_panel(results, search_time=elapsed)
            else:
                messagebox.showinfo("Info", "No results found")

        ctk.CTkButton(container, text="Search", command=run_query).pack(pady=10)

    def show_multi_query(self):
        self.clear_search_tab()
        container = self.tabview.tab("Search Results")
        ctk.CTkLabel(
            container, text="Multi Query (Text)",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=10)
        txt = ctk.CTkTextbox(container, height=220)
        txt.pack(fill="both", padx=20, pady=10)
        txt.insert("1.0", "Enter one query per line…")
        topk_entry = ctk.CTkEntry(container, placeholder_text="Top-K per query (default 100)")
        topk_entry.pack(pady=6, padx=20)

        def run_multi():
            raw     = txt.get("1.0", "end").strip()
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
                all_results.extend(self.retriever.search_text(q, top_k=topk, save_to_db=True))
            elapsed = time.time() - start
            if all_results:
                self._build_results_panel(all_results, search_time=elapsed)
            else:
                messagebox.showinfo("Info", "No results found")

        ctk.CTkButton(container, text="Run Multi-Query", command=run_multi).pack(pady=10)

    def show_image_query(self):
        self.clear_search_tab()
        container  = self.tabview.tab("Search Results")
        ctk.CTkLabel(
            container, text="Image Query",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=10)
        topk_entry = ctk.CTkEntry(container, placeholder_text="Top-K (default 100)")
        topk_entry.pack(pady=6, padx=20)
        preview    = ctk.CTkLabel(container, text="No image selected")
        preview.pack(pady=10)
        selected   = {"path": None}

        def pick_image():
            p = filedialog.askopenfilename(
                title="Choose an image",
                filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
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

        def run_img_query():
            if not selected["path"]:
                messagebox.showwarning("Warning", "Please select image")
                return
            try:
                topk = int(topk_entry.get()) if topk_entry.get().strip() else 100
            except ValueError:
                messagebox.showerror("Error", "Top-K must be integer")
                return
            start   = time.time()
            results = self.retriever.search_image(selected["path"], top_k=topk, save_to_db=True)
            elapsed = time.time() - start
            if results:
                self._build_results_panel(results, search_time=elapsed)
            else:
                messagebox.showinfo("Info", "No results found")

        ctk.CTkButton(container, text="Choose Image…", command=pick_image).pack(pady=6)
        ctk.CTkButton(container, text="Search by Image", command=run_img_query).pack(pady=6)

    # ── TRAKE panel ───────────────────────────────────────────────────────────

    def show_trake(self):
        """
        TRAKE panel.
        FIX: original called self.retriever.temporal_retrieval() which does not exist.
        Now calls self.retriever.trake_highest() (the correct method).
        """
        self.clear_search_tab()
        container = self.tabview.tab("Search Results")
        ctk.CTkLabel(
            container, text="TRAKE (Temporal Retrieval)",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=10)
        ctk.CTkLabel(container, text="Enter events (one per line) in temporal order:").pack(pady=4)

        txt = ctk.CTkTextbox(container, height=260)
        txt.pack(fill="both", padx=20, pady=10)
        txt.insert("1.0", "E1: …\nE2: …\nE3: …")

        topk_entry  = ctk.CTkEntry(container, placeholder_text="Top-K per event (default 200)")
        topk_entry.pack(pady=6, padx=20)
        topT_entry  = ctk.CTkEntry(container, placeholder_text="Top-T solutions (default 3)")
        topT_entry.pack(pady=6, padx=20)
        name_entry  = ctk.CTkEntry(container, placeholder_text="Output CSV name (without .csv)")
        name_entry.pack(pady=6, padx=20)

        def run_trake():
            raw    = txt.get("1.0", "end").strip()
            events = [l.strip() for l in raw.splitlines() if l.strip()]
            if not events:
                messagebox.showwarning("Warning", "Please enter at least one event")
                return
            try:
                topk = int(topk_entry.get()) if topk_entry.get().strip() else 200
            except ValueError:
                messagebox.showerror("Error", "Top-K must be integer")
                return
            try:
                topT = int(topT_entry.get()) if topT_entry.get().strip() else 3
            except ValueError:
                messagebox.showerror("Error", "Top-T must be integer")
                return

            start = time.time()
            # FIX: was temporal_retrieval() — correct method is trake_highest()
            sols  = self.retriever.trake_highest(events, top_k=topk, candidates=topT)
            elapsed = time.time() - start

            if not sols:
                messagebox.showinfo("No match", "Could not find any ordered sequence.")
                return
            if isinstance(sols, dict):
                sols = [sols]

            # Summary tab
            summary_tab = f"TRAKE:Top{len(sols)}"
            if summary_tab in self.tabview._tab_dict:
                try:
                    self.tabview.delete(summary_tab)
                except Exception:
                    pass
            self.tabview.add(summary_tab)
            self.tabview.set(summary_tab)
            panel_sum = self.tabview.tab(summary_tab)
            ctk.CTkLabel(
                panel_sum,
                text=f"Found {len(sols)} sequences | Search time: {elapsed:.2f}s",
                font=ctk.CTkFont(size=14, weight="bold")
            ).pack(pady=6)

            # Export-all button
            export_name_entry = ctk.CTkEntry(panel_sum, placeholder_text="Base CSV name (optional)")
            export_name_entry.pack(pady=4)

            def export_all():
                base = (export_name_entry.get() or "").strip() or f"trake_top{len(sols)}_{int(time.time())}"
                for i, s in enumerate(sols, 1):
                    try:
                        # Build flat payload expected by export_trake
                        payload = [
                            {
                                "video_name": s["video_name"],
                                "frame_idx":  fid,
                                "event":      len(s["frame_ids"]),
                            }
                            for fid in s["frame_ids"]
                        ]
                        self.retriever.export_trake(payload, f"{base}__rank_{i}")
                    except Exception as e:
                        messagebox.showerror("Export error", str(e))
                        return
                messagebox.showinfo("Export", f"Saved to ./submission/{base}__rank_*.csv")

            ctk.CTkButton(panel_sum, text="📄 Export ALL", command=export_all).pack(pady=6)

            # Per-solution tabs
            for i, sol in enumerate(sols, 1):
                vid     = sol["video_name"]
                details = sol["details"]
                tabname = f"TRAKE #{i}: {vid} ({sol.get('score', 0):.3f})"
                if tabname in self.tabview._tab_dict:
                    try:
                        self.tabview.delete(tabname)
                    except Exception:
                        pass
                self.tabview.add(tabname)
                panel = self.tabview.tab(tabname)
                ctk.CTkLabel(
                    panel,
                    text=f"[#{i}] Video: {vid} | Score: {sol.get('score', 0):.3f}",
                    font=ctk.CTkFont(size=13, weight="bold")
                ).pack(pady=6)

                for d in details:
                    try:
                        img = Image.open(d["path"]).convert("RGB")
                        img.thumbnail((240, 240))
                        tkimg = ImageTk.PhotoImage(img)
                        row   = ctk.CTkFrame(panel, corner_radius=8)
                        row.pack(fill="x", padx=10, pady=6)
                        lbl   = ctk.CTkLabel(row, image=tkimg, text="")
                        lbl.image = tkimg
                        lbl.pack(side="left", padx=8, pady=8)
                        lbl.bind("<Button-1>", lambda e, v=vid: self._open_video_frames_tab(v))
                        info = f"Event: {d['event']}\nFrame: {d['frame']}\nScore: {d['sim']:.3f}"
                        ctk.CTkLabel(row, text=info, anchor="w").pack(side="left", padx=10)
                    except Exception as e:
                        print(f"⚠️ Load error {d.get('path')}: {e}")

        ctk.CTkButton(container, text="Run TRAKE", command=run_trake).pack(pady=8)

    # ── Video frame browser tab ───────────────────────────────────────────────

    def _open_video_frames_tab(self, video_name: str):
        frames = self.retriever.get_frames_of_video(video_name)
        if not frames:
            messagebox.showinfo("Info", f"No frames found for video {video_name}")
            return

        tabname = f"Frames:{video_name}"
        if tabname in self.tabview._tab_dict:
            try:
                self.tabview.delete(tabname)
            except Exception:
                pass
        self.tabview.add(tabname)
        self.tabview.set(tabname)
        panel = self.tabview.tab(tabname)

        canvas     = ctk.CTkCanvas(panel, bg="#1f1f1f", highlightthickness=0)
        vsb        = ctk.CTkScrollbar(panel, orientation="vertical", command=canvas.yview)
        list_frame = ctk.CTkFrame(canvas)
        list_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=list_frame, anchor="nw")
        canvas.configure(yscrollcommand=vsb.set)
        canvas.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        thumbs = []
        self.tab_thumbs[tabname] = thumbs
        for r in frames:
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
                    anchor="w"
                ).pack(side="left", padx=8)
            except Exception as e:
                print(f"⚠️ Error loading frame {r.get('image_path')}: {e}")

    # ── Other panels ──────────────────────────────────────────────────────────

    def show_stored_queries(self):
        self.clear_search_tab()
        container = self.tabview.tab("Search Results")
        ctk.CTkLabel(
            container, text="Stored Queries",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=10)
        queries = self.retriever.get_stored_queries()
        if not queries:
            ctk.CTkLabel(container, text="No stored queries",
                         font=ctk.CTkFont(size=12)).pack(pady=20)
            return
        list_frame = ctk.CTkFrame(container, corner_radius=8)
        list_frame.pack(fill="both", expand=True, padx=10, pady=10)
        for i, q in enumerate(queries):
            def make_cmd(idx=i):
                return lambda: self._load_and_show_query(idx)
            ctk.CTkButton(list_frame, text=f"[{i}] {q}", command=make_cmd()).pack(pady=4, fill="x")

    def _load_and_show_query(self, idx: int):
        results = self.retriever.reload_query_results(idx)
        if results:
            self.tabview.set("Search Results")
            self._build_results_panel(results, search_time=None)
        else:
            messagebox.showinfo("Info", "Invalid query index or no results")

    def export_database(self):
        self.retriever.export_database("./exported_vectors")
        messagebox.showinfo("Export", "Database exported to ./exported_vectors_[embeddings|metadata].*")

    def show_stats(self):
        self.clear_search_tab()
        s    = self.retriever.get_stats()
        info = (
            f"Total images: {s['total_images']}\n"
            f"Total videos: {s['total_videos']}\n"
            f"Embedding shape: {s['embedding_shape']}"
        )
        ctk.CTkLabel(
            self.tabview.tab("Search Results"),
            text=info,
            font=ctk.CTkFont(size=13)
        ).pack(pady=20)


if __name__ == "__main__":
    app = EVA02GUI()
    app.mainloop()
