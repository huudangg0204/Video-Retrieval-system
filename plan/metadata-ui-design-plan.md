# Kế hoạch Thiết kế Tính năng Ứng dụng Lifelog (AIC-26)

Dựa trên các bài báo khoa học từ Lifelog Search Challenge (LSC 2024 & 2025) như **lifeXplore**, **SnapSeek 3.0**, và **MemoriEase 3.0**, cùng với hệ thống metadata phong phú mà chúng ta đã trích xuất (Caption, OCR, Detections, Scene, Scene Graph), dưới đây là kế hoạch thiết kế các tính năng cho ứng dụng (GUI/Frontend) nhằm tối đa hóa hiệu quả truy vấn.

## 1. Multi-modal Query Builder (Inspired by lifeXplore & SnapSeek)
Giao diện tìm kiếm không chỉ là một thanh search đơn thuần mà là một **Query Builder** cho phép kết hợp linh hoạt free-text và metadata filters.

*   **Free-text Search**: Đầu vào chính, sử dụng cơ chế multi-stream (EVA-02 + BLIP-2 + BM25 Caption) đã xây dựng.
*   **Metadata Filters Component**:
    *   **Text/OCR Filter**: Ô nhập liệu riêng để truy vấn chính xác đoạn text xuất hiện trong ảnh (tận dụng trường `ocr_text`). Hữu ích cho các query dạng "đang đọc thực đơn quán X", "biển báo đường Y".
    *   **Object Presence**: Dropdown/Tag selector cho phép chọn các object bắt buộc có hoặc không có trong khung hình (tận dụng trường `detections`).
    *   **Scene / Location Context**: Dropdown chọn ngữ cảnh không gian thông qua `scene_group` (VD: Indoor, Outdoor, Vehicle) hoặc cụ thể hơn qua `smoothed_scene` (VD: coffee_shop, bedroom).
    *   **Scene Graph (Quan hệ không gian/hành động)**: Giao diện dạng `[Chủ thể] -> [Hành động] -> [Đối tượng]`. Người dùng có thể tìm kiếm dựa trên các bộ ba (triplet) đã trích xuất, VD: `Person -> holding -> cup`. (SnapSeek 3.0 feature).

## 2. Temporal & Event Sequence Search (Inspired by SnapSeek & lifeXplore)
Hỗ trợ các câu truy vấn chuỗi thời gian (KIS task) thường thấy trong LSC khi thông tin được cung cấp dần dần theo thời gian.

*   **Sequence Input Builder**: Cho phép người dùng thêm các "bước" của sự kiện. VD: `[Bước 1: Đi bộ ở công viên] -> [Bước 2: Lái xe] -> [Bước 3: Ăn tại nhà hàng]`. Backend sẽ sử dụng tính năng **TRAKE** (Temporal Retrieval) để tìm kiếm chuỗi khớp nhất.
*   **Contextual Exploration ("Explore around this time")**: Khi click vào 1 bức ảnh, UI cung cấp nút xem trước và sau thời điểm đó (tận dụng các trường `prev_frame_idx`, `next_frame_idx`) để kiểm tra bối cảnh (context) của khoảnh khắc.

## 3. Conversational QA & RAG (Inspired by MemoriEase 3.0)
Tích hợp một trợ lý ảo (Chatbot) vào giao diện để giải quyết các task dạng Question-Answering (QA).

*   **Natural Language QA**: Người dùng có thể hỏi "Tôi đã đi siêu thị bao nhiêu lần vào tháng trước?". 
*   **RAG Pipeline**: Ứng dụng sẽ dùng metadata (vd: filter `scene_group == shopping`) để query các keyframe liên quan, đếm số lượng event (dựa trên nhóm thời gian gần nhau), sau đó đưa cho LLM tổng hợp lại thành câu trả lời tự nhiên cho người dùng.

## 4. Interactive Relevance Feedback (Inspired by MemoriEase 3.0)
Hỗ trợ người dùng "dạy" hệ thống trong thời gian thực khi đang tìm kiếm.

*   **Nút Thích/Không thích (Thumbs Up/Down)**: Trên mỗi kết quả trả về, thêm nút Like/Dislike.
*   **Dynamic Re-ranking**: Khi người dùng tương tác, frontend gửi feedback về backend để backend điều chỉnh vector query ban đầu (cộng thêm embedding của ảnh Like, trừ đi embedding của ảnh Dislike) và cập nhật lại kết quả ngay lập tức mà không cần nhập lại text.

## 5. Advanced Result Visualization (Inspired by SnapSeek 3.0)
Trình bày kết quả trực quan giúp duyệt (browse) hàng ngàn ảnh nhanh chóng.

*   **Cluster View / Concept Grouping**: Nhóm các kết quả trả về theo `scene_group` (VD: Các kết quả ở "Ngoài trời" xếp chung 1 nhóm, "Trong nhà" chung 1 nhóm) để người dùng dễ dàng thu hẹp phạm vi mắt nhìn.
*   **Metadata Inspector**: Khi hover/click vào 1 bức ảnh, hiển thị overlay một panel chi tiết liệt kê toàn bộ: Bounding box của objects, đoạn text OCR được highight, Scene category, và Caption. Giúp người dùng xác nhận nhanh độ chính xác của ảnh.
*   **Grid Size Toggle**: Cho phép phóng to/thu nhỏ lưới ảnh linh hoạt để duyệt nhanh (fast navigation).

---

## 🚀 Đề xuất Lộ trình Triển khai (Roadmap)
1.  **Phase 1 (Cơ bản)**: Hoàn thiện UI cho thanh Free-text Search kết hợp Metadata Filters (Scene, Object, OCR) panel. Hiển thị Metadata Inspector khi click vào ảnh.
2.  **Phase 2 (Trung cấp)**: Tích hợp UI cho Temporal Search (TRAKE) và nút "Explore around this time". Thêm tính năng Relevance Feedback (Like/Dislike).
3.  **Phase 3 (Nâng cao)**: Xây dựng cửa sổ Chat (Conversational QA) sử dụng RAG. Triển khai Cluster View để gom nhóm kết quả. Tích hợp Scene Graph filter phức tạp.
