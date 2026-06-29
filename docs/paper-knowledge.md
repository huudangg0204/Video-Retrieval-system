# Paper Knowledge Library — AIC-26 Research Foundation

Knowledge library tổng hợp từ 5 bài báo nền tảng trong `paper/`. Đây là các paper tham khảo trực tiếp cho hệ thống AIC-26: chúng định hình các quyết định kiến trúc (dual-embedding EVA-02 + BLIP-2, TRAKE DP, OCR + object filter, RAG cho QA) và là baseline benchmark khi đánh giá retrieval quality.

---

## Tổng quan 5 paper

| # | Paper | Năm | Hội nghị | Vai trò đối với AIC-26 |
|---|---|---|---|---|
| 1 | [The State-of-the-Art in Lifelog Retrieval](#1-lsc-review--state-of-the-art-trong-lifelog-retrieval-2022-2024) | 2025 | IEEE Access | **Survey** — bản đồ hóa toàn bộ paradigm retrieval 2022-24, benchmark & metric. Định hướng lý thuyết. |
| 2 | [lifeXplore at LSC 2024](#2-lifexplore-tại-lsc-2024) | 2024 | ACM ICMR / LSC'24 | **Đối thủ thắng LSC'23** — kiến trúc CLIP + FAISS + MongoDB, OpenCLIP ViT-L/14 (LAION2B). Reference cho CLIP-only pipeline. |
| 3 | [Libro — Lifelog Search Browser](#3-libro--lifelog-search-browser) | 2024 | ACM ICMR / LSC'24 | **Keyframe + Shot detection**, ANNS trên proximity graph, filter mask Roaring Bitmaps, EVA-CLIP ViT-E fine-tune. Reference cho keyframe extraction. |
| 4 | [MemoriEase 3.0 — RAG-Enhanced Conversational](#4-memoriease-30--rag-enhanced-conversational-retrieval) | 2025 | ACM ICMR / LSC'25 | **Dual-embedding CLIP + BLIP-2** chính là nền tảng cho cơ chế dual-stream trong AIC-26. Elasticsearch + cross-encoder rerank + RAG. **Quan trọng nhất.** |
| 5 | [SnapSeek 3.0 — Enhanced Scene Understanding](#5-snapseek-30--enhanced-scene-understanding) | 2025 | ACM ICMR / LSC'25 | **Temporal search nâng cao**: SeqWin, ParChain, scene graph 3-tuple qua MLLM (Qwen2.5-VL-7B), ADL (Activities of Daily Living). Reference cho TRAKE. |

---

## 1. LSC Review — State-of-the-Art trong Lifelog Retrieval 2022-2024

**File**: `paper/2506.06743v2.pdf` — IEEE Access, Dec 2025.
**Tác giả**: Tran et al. (DCU, Klagenfurt, Reykjavik, etc.).

### Điểm chính

LSC (Lifelog Search Challenge) là benchmark dài hạn duy nhất đánh giá hệ thống retrieval tương tác real-time qua human-in-the-loop. Khác với TRECVID / VBS / ImageCLEF (offline asynchronous), LSC chạy live với assessors.

### Dataset

- **725K ảnh first-person** từ Narrative Clip, 1024×768, 18 tháng (Jan 2019 – Jun 2020)
- Metadata mỗi phút: timestamp, activity, biometrics (heart-rate, steps, sleep), GPS
- Visual concepts: COCO objects (80 classes) + Places365 scenes (102 labels)
- OCR + bounding box + confidence
- Semantic location names ("Home", "Dublin City University")
- Flight metadata (Voxento team)

### 3 loại task

| Task | Query | Time | Score |
|---|---|---|---|
| **KIS** (Known-Item Search) | text mô tả progressing qua các hint 30s | 300s | `max(0, 100 − 50·t/T − 10·w)` |
| **Ad-hoc** | free-text, không có ground truth | 180s | `100 × correct/(correct+incorrect/2) × correct/total` |
| **QA** | câu hỏi tự do → text answer | 180s | Correctness judged by assessors, penalty cho incorrect |

### Evolution của hệ thống

- **LSC'22**: CLIP zero-shot lên ngôi (MyScéal, LifeSeeker 4.0, Memento 2.0, Vitrivr-VR, EAGLE, Voxento, FIRST, vitrivr).
- **LSC'23**: text-image embedding trên hầu hết các hệ thống; temporal + spatial filtering phổ biến.
- **LSC'24**: tích hợp metadata, concept, temporal reasoning; Llama-based reranking.

### Paradigm shift

```
[2018-2021] rule-based / keyword filter
   ↓
[2022+] embedding-driven (CLIP family) + FAISS vector index
   ↓
[2024+] LLM-driven retrieval + RAG (MemoriEase 3.0) + multimodal (image+text+sketch)
```

### Bài học rút ra cho AIC-26

1. **Multimodal input handling** quan trọng — combine text query + image + metadata filters.
2. **Temporal reasoning** (TRAKE-style DP) đã thành standard cho KIS/QA.
3. **Real-time interaction** (sub-second response) là yếu tố cạnh tranh.
4. **Object filter + OCR + semantic location** là ba filter "must-have".
5. **Score formula phạt wrong submissions** — nên có UI giúp user refine, không spam.

---

## 2. lifeXplore tại LSC 2024

**File**: `paper/3643489.3661123.pdf` — ACM ICMR / LSC'24.
**Tác giả**: Martin Rader & Klaus Schoeffmann (Klagenfurt University).
**Đây là winner LSC'23**, cải tiến cho LSC'24.

### Architecture 3 lớp

```
┌──────────────────────────────────────────┐
│  Backend: image analysis + metadata       │
│  (YOLOv7, EfficientNet B2, CRAFT, OpenCLIP)│
└──────────────────┬───────────────────────┘
                   │ JSON over HTTP
┌──────────────────▼───────────────────────┐
│  Middleware: Python + Node.js             │
│  - FAISS (CLIP features)                  │
│  - MongoDB (metadata + bounding box)      │
│  - Combined query handler                 │
└──────────────────┬───────────────────────┘
                   │ WebSocket
┌──────────────────▼───────────────────────┐
│  Frontend: Angular GUI                    │
│  - Query blocks (special commands: -o, -c)│
│  - Temporal layout                        │
│  - Detail dialog                          │
└───────────────────────────────────────────┘
```

### Models được dùng

| Modality | Model | Dim |
|---|---|---|
| Semantic Concepts | EfficientNet B2 (Places365, 365 categories) | — |
| Objects | YOLOv7 (80 MS COCO categories) | — |
| Text (OCR) | CRAFT + EasyOCR | — |
| Text Embedding | OpenCLIP ViT-L/14 (LAION2B) | 768 |
| Image Embedding | OpenCLIP ViT-L/14 (LAION2B) | 768 |

Cả text và image dùng cùng OpenCLIP model → shared embedding space → cosine similarity trực tiếp.

### Special query commands (prefix `-`)

| Command | Modality |
|---|---|
| `-m` | Month |
| `-y` | Year |
| `-wd` | Weekday |
| `-o` | Detected object (COCO label) |
| `-c` | Detected concept (Places365) |
| `-t` | Recognized text (OCR) |
| `-h` | Heart rate (e.g. `-h 90+-10` = 80-100) |
| `-country`, `-city` | Geo-name |

### Sub-filter (prefix `|`)

- `|position:top-left|`, `|bottom-right|` — bounding box position
- `|score:0.8|` — min confidence

### Temporal search

Query phức tạp kiểu `"man with red shirt -o person|position:top-left < plane flight <-h 120+"` được tách thành nhiều block. Mỗi block tìm 5000 FAISS results → lọc qua MongoDB bounding box → ghép theo thời gian.

Heart-rate filter tự động kích hoạt khi trong query có giá trị số > 120.

### Kết quả Recall@K (text embedding benchmark)

OpenCLIP ViT-L/14 (LAION2B) đạt Recall@1 = 0.7, @10 = 0.8, @100 = 0.9 trên LSC'2023 textual queries. Tốt nhất trong các model được thử.

### Bài học rút ra cho AIC-26

1. **OpenCLIP ViT-L/14 (LAION2B) là baseline mạnh** — AIC-26 dùng EVA-02 (1024-dim) nên phong phú hơn về dim.
2. **Image-as-Query** cho phép user cung cấp thêm hint qua URL/upload/drag — AIC-26 đã có (`/api/upload_image`).
3. **Temporal query composition** bằng cách chia thành nhiều query part rồi ghép theo thời gian — giống cơ chế TRAKE trong AIC-26 nhưng khác ở chỗ: lifeXplore ghép qua MongoDB key-time, AIC-26 dùng DP.
4. **Bounding box + position filter** — YOLOv7 COCO 80 categories. AIC-26 dùng YOLOv8x OIV7 (600+ categories) phong phú hơn nhiều.
5. **WebSocket cho frontend** thay vì polling — AIC-26 dùng HTTP REST (Flask), đơn giản hơn nhưng có thể cải tiến.

---

## 3. Libro — Lifelog Search Browser

**File**: `paper/3643489.3661124.pdf` — ACM ICMR / LSC'24.
**Tác giả**: Hezel, Schall, Schilling, Jung, Barthel (HTW Berlin).
**Tiền thân**: hệ thống *Vibro* trong VBS (Video Browser Showdown) — chuyển từ video sang lifelog.

### Pipeline

```
FRAMES (725,226) → KEYFRAMES (397,101) → SHOTS (~205K)
                              ↓                ↓
                     L2 feature distance   Roaring Bitmaps
                     (consecutive frames    (weekday filter)
                      merged if similar)
```

### Keyframe + Shot Detection

- Tính L2 distance giữa feature vector của các frame liên tiếp.
- Nếu nhỏ hơn threshold → merge thành 1 keyframe (lấy highest contrast).
- Sau khi có keyframes → so sánh keyframe đầu tiên của shot với keyframe cuối của shot trước → chia shot.
- 725K frames → 397K keyframes → ~205K shots.

### Search capabilities

- **Text search**: CLIP offline — feature server chạy GPU, query online trên CPU.
- **Image search**: upload/drag → server tạo embedding → ANN search.
- **OCR**: BM25 + alpha slider (0 = image, 1 = OCR text).
- **Hybrid**: combine text + image + OCR với slider weights.

### Models đã thử

| Model | Dim | Note |
|---|---|---|
| CLIP (original) | 1024 | Joint-embedding, contrastive |
| LAIONSB | — | CLIP loss + auxiliary loss |
| EVA CLIP ViT-E | — | **5B params, lớn nhất vision-language open-source** |

Best VBS: **EVA CLIP ViT-E** joint-embedding + LAIONSB loss → quantized to 1536-dim + binarize ở range [0, 255] → dùng **Hamming distance** thay L2/cosine → memory chỉ 192 bytes/image. Tốc độ tăng ~100×.

### Roaring Bitmap filters

Filter mask cho mỗi metadata field:
- Weekday filter (Sat+Sun UNION).
- Biometric (heart-rate range).
- Location (city/country).
- Visual concept (Places365).
- Date range.

Mỗi filter → Roaring Bitmap → AND operation trong milliseconds.

### User Interface

3 panels:
1. **Query formulation** (trái): text/image/OCR input + sliders + filters.
2. **Search area** (giữa): scan-line grid (FLAS algorithm) — most relevant ở giữa.
3. **Global map** (phải trên): 2D grid visualization của proximity graph — clustering tự nhiên.
4. **Temporal view** (phải dưới): slideshow ngày của ảnh hiện tại.

### Bài học rút ra cho AIC-26

1. **Keyframe extraction giảm search space 50%** — AIC-26 dùng DINOv2 cosine similarity cho keyframe detection (`pipeline/keyframe.py`).
2. **Quantized embedding + Hamming** tăng tốc 100× so với L2/cosine — cơ hội tối ưu hóa cho AIC-26 nếu scale lên hàng triệu frames.
3. **EVA CLIP ViT-E** joint-embedding với LAIONSB loss → AIC-26 cũng dùng **EVA-02** nhưng chưa fine-tune joint-embedding.
4. **Roaring Bitmap cho filter** — nhanh hơn so với query DB mỗi lần.
5. **FLAS algorithm** cho scan-line grid arrangement → có thể áp dụng cho hiển thị kết quả GUI.

---

## 4. MemoriEase 3.0 — RAG-Enhanced Conversational Retrieval

**File**: `paper/3729459.3748689.pdf` — ACM ICMR / LSC'25.
**Tác giả**: Tran, Nguyen, Jones, Gurrin (DCU + VNU-HCM).
**Đây là paper nền tảng cho kiến trúc dual-embedding của AIC-26** — đặc biệt là §3.2 và cơ chế "weighted average of CLIP + BLIP2 cosine similarity".

### Tại sao paper này quan trọng

AIC-26 có `USE_DUAL_EMBEDDING=true` + `data/blip2_embeddings/` → load 2 stream embedding EVA-02 + BLIP-2 Q-Former. Đây chính xác là cách MemoriEase 3.0 làm ở §3.2 — weighted average hai embedding.

### Kiến trúc tổng quan

```
            OFFLINE                              ONLINE
┌────────────────────────────┐         ┌──────────────────────┐
│ Metadata CSV               │         │ User → Chat UI       │
│ BLIP2 image descriptions   │ Indexing│   ↓                  │
│ Images (lifecycle)         │─────────▶ Elasticsearch       │
│                            │         │   ↓                  │
│ Embedding Models:          │         │ Embedding Models     │
│  - BLIP2 (256-dim)         │         │ (CLIP + BLIP2)       │
│  - CLIP (1024-dim)         │         │   ↓                  │
└────────────────────────────┘         │ Top-K images         │
                                       │   ↓                  │
                                       │ Reranking (BERT)     │
                                       │   ↓                  │
                                       │ GPT-o1 → answer      │
                                       └──────────────────────┘
```

### Dual embedding §3.2 (CRITICAL cho AIC-26)

> "We extract image embeddings using the **BLIP2 and CLIP** models. [...] query is encoded into embeddings by **two models (CLIP and BLIP2)**. The embeddings are used to calculate the cosine similarity from image embeddings. We use a **weighted average approach** to combine the cosine similarity from two embeddings. The weight is chosen for retrieving several queries and tuning the best weight."

→ Đây chính là `EVA02_SEARCH_WEIGHT * sim_eva02 + BLIP2_SEARCH_WEIGHT * sim_blip2` trong `core/search.py::_dual_topk_indices` của AIC-26.

### Kích thước embedding

| Model | Dim |
|---|---|
| BLIP2 | 256 |
| CLIP | 1024 |
| EVA-02 (AIC-26 dùng) | 768 |

### Các cải tiến so với MemoriEase 2.0

1. **Dual embedding (CLIP + BLIP2)** — chính là core của §3.2 mô tả.
2. **Enhanced Relevance Feedback** — trước: dùng pooled image average. Sau: dùng query embedding + pooled image embedding với weight balance để retain original query info.
3. **RAG cho QA task** — cross-encoder BERT rerank descriptions, format `<metadata><description>` prompt cho GPT-o1.
4. **KIS score cải thiện 10 điểm** — nhờ query embedding retention.
5. **AD-HOC không cải thiện** — vì cùng cơ chế visual similarity search.

### BLIP2 image descriptions

- Sinh caption tự động từ BLIP2 (Salesforce/blip2-itm-vit-g).
- Lưu thêm vào index → filter/re-rank theo description relevance.
- ⚠️ **AIC-26 gotcha**: `Salesforce/blip2-itm-vit-g` chỉ nhận image input cho Q-Former, text phải dùng black 224×224 placeholder (xem memory [[blip2-itm-vit-g-image-only]]). Đã có runtime probe fallback trong `core/search.py` (xem [[dual-embedding-text-fallback]]).

### Results @ LSC'25

- **3rd place overall** (snapSeek 1st, 2nd, MemoriEase 3rd)
- **2nd place QA task** (score 96/100)
- 7/10 KIS đúng, 6/6 QA đúng, kém Ad-hoc

### Bài học rút ra cho AIC-26

1. **§3.2 weighted average hai embedding** — đã implement trong AIC-26, kiểm chứng giả thuyết từ paper này.
2. **Pooled image embedding ≠ original query info** — phải weight lại với query embedding (relevance feedback cải tiến).
3. **BLIP2 Q-Former image-only** — phải fallback cho text-only query (memory [[blip2-itm-vit-g-image-only]]).
4. **Cross-encoder BERT rerank** — AIC-26 dùng BLIP ITM rerank thay vì BERT. Khác nhưng cùng paradigm.
5. **RAG cho QA** — nếu AIC-26 mở rộng sang QA task, có thể mượn cơ chế `description → prompt → GPT` này.

---

## 5. SnapSeek 3.0 — Enhanced Scene Understanding

**File**: `paper/3729459.3748697.pdf` — ACM ICMR / LSC'25.
**Tác giả**: Ho-Le, Ho, Ninh, Nguyen, Gurrin, Tran (VNU-HCM + DCU).
**Top performer LSC'25** (2nd place overall, 1st place KIS — score 100/100).

### 4 trụ cột cải tiến từ SnapSeek 2.0

1. **Search modalities**: semantic, image-based, keyword, hybrid.
2. **Temporal search nâng cao**: SeqWin + ParChain.
3. **Human-in-the-loop feedback** với A+ / A− update.
4. **Scene graph 3-tuple** qua MLLM (Qwen2.5-VL-7B).
5. **ADL (Activities of Daily Living)** annotations — 35 classes.

### Search modalities

- **Semantic**: CLIP5S text-image → Milvus dense vector.
- **Image-based**: upload/URL/cropped → CLIP image embedding.
- **Keyword**: location/ADL/OCR indexed as sparse vector trong Milvus.
- **Hybrid**: dense + sparse weighted fusion.

→ AIC-26 chỉ có semantic + image + object filter. Keyword/ADL sparse search là cơ hội mở rộng.

### Temporal search — 4 bước

#### Step 1: Temporal Clause Extraction (LLM)

LLM prompt-engineered convert text query → JSON:
```json
{
  "before": <clause_1>,
  "after":  <clause_2>,
  "interval": {"min": x, "max": y}  // null nếu không có
}
```

Cho phép `null interval` → linh hoạt cho "arrive at the airport before boarding" (gap không xác định).

#### Step 2: Clause Retrieval

Mỗi clause → search stack semantic → top-K ranked lists R1, R2.

#### Step 3: Temporal Consistency Enforcement

**Hai mode interchangeable:**

##### SeqWin (Sequential Windowing)
Cho tightly-coupled temporal queries.

```
1. Lấy top-K seed frames f1 từ R1 (ranked by sim s1).
2. Với mỗi f1 tại time t_f1, tạo forward window W(f1) = [t_f1, t_f1 + δ]
   với δ = interval.max hoặc user-adjustable.
3. Trong W(f1), xét tất cả frames f2 trong R2 có timestamp ∈ W(f1)
   → tính clause-2 score s2 → chọn frame gần f1 nhất (smallest gap).
4. pair_score = mean(s1, s2)
5. Rank các valid (f1, f2) pairs.
```

→ Tương đương `trake_closest` (greedy DP) trong AIC-26 — minimize frame gap.

##### ParChain (Parallel Chain Aggregation)
Cho loosely-coupled / multi-scene queries.

```
1. Từ R1 và R2, form contexts — chuỗi frame kề nhau trong interval [min, max].
   context = shot (≤ 2s window).
2. score(c) = max(f) trong context.
3. Tạo ordered context pairs (c1, c2) với c1 ∈ R1, c2 ∈ R2
   sao cho start(c2) > end(c1) và gap ∈ [min, max].
4. pair_score = sum(score(c1), score(c2))
5. Rank valid context pairs.
```

→ Tương đương `trake_highest` (summed similarity) trong AIC-26.

#### Step 4: Interactive Refinement

Slider α ∈ [1..200] re-evaluate SeqWin/ParChain on-the-fly mà không phải re-index.

### Human Feedback

User mark ảnh relevant / irrelevant. Cập nhật set A:

```
A ← (A ∪ A⁺) \ A⁻
```

Mỗi feedback có ít nhất 1 component: relevant set, irrelevant set, hoặc cả hai. Hai-scheme scoring:
- **A⁺ (positive)**: similarity = α × original_score + (1−α) × sim_to_relevant
- **A⁻ (negative)**: similarity = original_score × sim_to_irrelevant

### Scene Graphs (NEW in 3.0)

3-tuple representation:
```
(subject, predicate, object)
```

Generated bằng **Qwen2.5-VL-7B** MLLM với carefully-crafted prompt (lower-case noun, base-form predicates, no passive constructions). Indexed in Milvus alongside image embeddings.

→ Enable both visual + relational search. AIC-26 chưa có scene graph — cơ hội cải tiến lớn.

### ADL (Activities of Daily Living)

- 35 classes (e.g., "cooking", "commuting", "shopping").
- 300 representative examples/class.
- High-quality annotation ≈ 75% dataset.
- Used để filter/narrow retrieval candidates cho action-related queries.

→ AIC-26 có object detection (YOLOv8x OIV7) nhưng chưa có ADL classification — mở rộng tự nhiên.

### Results @ LSC'25

- **2nd place overall** (SnapSeek1: 2711/3000 = 90.4%)
- **1st place KIS** với SnapSeek2 (100% on some queries)
- Top 3 trên Ad-hoc

### Bài học rút ra cho AIC-26

1. **SeqWin ≡ trake_closest, ParChain ≡ trake_highest** — AIC-26 đã implement nhưng còn đơn giản. Có thể nâng cấp:
   - Explicit interval [min, max] parsing (dùng LLM hoặc regex).
   - Forward window thay vì exhaustive DP (nhanh hơn).
   - Context = shot (window theo shot, không phải per-frame).
2. **α slider re-evaluate on-the-fly** — đáng thêm vào GUI: refine temporal results mà không cần re-search.
3. **Scene graph 3-tuple qua MLLM** — feature chưa có trong AIC-26. Có thể tích hợp Qwen2.5-VL-7B nếu có GPU đủ mạnh (RTX 2080 Ti chỉ có 11GB → có thể chạy quantized).
4. **ADL classification** — bổ sung cho action queries. Pipeline tương tự YOLOv8x.
5. **Milvus thay FAISS** — nếu scale lên > 1M embeddings, Milvus có GPU index và distributed. Hiện tại AIC-26 dùng FAISS IndexFlatIP (CPU) đủ cho ~700K vectors.
6. **LLM clause extraction** — quan trọng cho query parsing. Nếu thêm LLM step (GPT-4o-mini, Llama-3) có thể boost temporal search accuracy.

---

## Ma trận so sánh kiến trúc AIC-26 với 5 paper

| Component | AIC-26 hiện tại | lifeXplore'24 | Libro'24 | MemoriEase 3.0 | SnapSeek 3.0 |
|---|---|---|---|---|---|
| **Vision encoder** | EVA-02 (768-dim) | OpenCLIP ViT-L/14 (768) | EVA CLIP ViT-E (1536 → quant 1024 binarize) | CLIP (1024) + BLIP2 (256) | CLIP5S |
| **Vector DB** | FAISS IndexFlatIP (CPU) | FAISS + MongoDB | FAISS + ANNS proximity graph | Elasticsearch | Milvus dense+sparse |
| **Object detection** | YOLOv8x OIV7 (600+ classes) | YOLOv7 (80 COCO) | — | — | CLIP implicit |
| **OCR** | EasyOCR (vi/en) | CRAFT | BM25 + alpha slider | — | sparse Milvus |
| **Temporal search** | TRAKE DP (greedy + summed) | Multi-block + MongoDB time-join | — (treat as continuous video) | — | SeqWin + ParChain |
| **Reranking** | BLIP ITM (`*` suffix trigger) | — | — | BERT cross-encoder | human feedback A+/A− |
| **RAG/QA** | ❌ | ❌ | ❌ | ✅ GPT-o1 | ❌ |
| **Scene graph** | ❌ | ❌ | ❌ | ❌ | ✅ Qwen2.5-VL-7B |
| **ADL classification** | ❌ | ❌ | ❌ | ❌ | ✅ 35 classes |
| **Relevance feedback** | ❌ (chỉ object filter) | ✅ via MongoDB query | ❌ | ✅ query+pooled embedding | ✅ A+/A− formula |
| **GUI framework** | Flask web UI + CustomTkinter | Angular | Native desktop | Chat UI | Web UI (Fuse.js fuzzy) |
| **Backend framework** | Flask | Node.js + Python | Python (Vibro base) | Python (Elasticsearch) | Python (Milvus) |
| **Lazy loading** | ✅ API server (on first request) | ❌ all at startup | — | — | — |

---

## Roadmap cải tiến tiềm năng cho AIC-26

Dựa trên các paper, đây là những cải tiến có giá trị cao:

### Ngắn hạn (low-hanging fruit)

1. **Temporal interval parsing**: parse `(min, max)` từ query text thay vì chỉ dùng `GAP_KEYFRAME` cố định (từ SnapSeek 3.0).
2. **Query embedding retention trong relevance feedback**: thay vì chỉ filter bằng object, mix pooled image embedding với query embedding (từ MemoriEase 3.0 §3.5).
3. **Binarized embedding cho large-scale**: nếu > 1M frames → quantize EVA-02 về [0,255] + Hamming (từ Libro).
4. **Hybrid sparse + dense search**: thêm BM25/keyword layer cho OCR text (từ SnapSeek 3.0).

### Trung hạn

5. **Scene graph generation**: dùng Qwen2.5-VL-7B quantized để sinh 3-tuple `(subject, predicate, object)` cho mỗi keyframe (từ SnapSeek 3.0 §3.5).
6. **Cross-encoder reranker**: thay BLIP ITM bằng BERT cross-encoder trên query-description pairs (từ MemoriEase 3.0 §3.4).
7. **RAG cho QA**: nếu mở rộng sang QA task, dùng GPT-4o-mini + reranked descriptions (từ MemoriEase 3.0 §3.4).

### Dài hạn

8. **Milvus thay FAISS**: distributed vector DB cho scale > 1M embeddings.
9. **ADL classification**: 35-class ADL pipeline (YOLOv8 + CLIP zero-shot) (từ SnapSeek 3.0 §3.6).
10. **CLIP-based joint-embedding fine-tune**: train EVA-02 với LAIONSB loss (joint embedding + binarize) — nhưng cần GPU > 24GB.

---

## Citation / tham chiếu nhanh

Khi implement feature mới dựa trên paper, cite trong commit message và docstring:

```python
# Ví dụ trong core/search.py
def dual_topk_indices(...):
    """Weighted average of two embedding streams.

    Implements the dual-embedding retrieval strategy from:
    MemoriEase 3.0 (Tran et al., LSC'25) §3.2:
        "weighted average approach to combine the cosine similarity
         from two embeddings (CLIP and BLIP2)"
    """
    score = EVA02_SEARCH_WEIGHT * sim_eva02 + BLIP2_SEARCH_WEIGHT * sim_blip2
```

---

## Tài liệu liên quan trong dự án

- `docs/dual-embedding.md` — implementation chi tiết dual-embedding cho AIC-26.
- `CLAUDE.md` §"Dual-stream search" — gotchas BLIP-2 Q-Former image-only.
- Memory: [[blip2-itm-vit-g-image-only]], [[dual-embedding-text-fallback]].

---

*Tạo ngày 2026-06-26 từ 5 PDF trong `paper/`.*
