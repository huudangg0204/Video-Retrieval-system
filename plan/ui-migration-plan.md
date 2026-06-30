# Kế hoạch Triển khai Giao diện (UI Migration Plan)

## Mục tiêu
Nâng cấp `index.html` từ giao diện cơ bản hiện tại lên thiết kế **Multi-modal Query Builder** theo `metadata-ui-design-plan.md` (Phase 1), đồng thời **bảo tồn 100% các tính năng cũ**.

## 1. Các tính năng CŨ cần giữ lại nguyên vẹn
1. **Text Search** (Free-text, objects, threshold, top-k).
2. **Text Search with Image** (Upload image để rerank).
3. **Image Search** (Tải ảnh lên để tìm kiếm tương đồng).
4. **TRAKE** (Temporal Retrieval, Top-K, Candidates, Events sequence).
5. **Video Frames Viewer & Range Viewer** (Xem ảnh liền kề theo Frame ID & Range).
6. **Export & Submission** (Export CSV, Group by video, Make Submission form).

## 2. Các tính năng MỚI cần thêm vào (Phase 1)
1. **Giao diện Query Builder**: 
   - Thanh search chia làm nhiều thẻ: `Text / OCR / Scene / Object`.
   - Bổ sung ô nhập `OCR Text` (để query riêng text xuất hiện trong ảnh).
   - Dropdown chọn `Scene Group` (Indoor, Outdoor, v.v.).
2. **Metadata Inspector Modal**:
   - Khi bấm vào một bức ảnh kết quả, mở ra một khung Modal thay vì chỉ mở tab ảnh mờ.
   - Hiển thị chi tiết: `Bounding Box (Detections)`, `OCR Text Highlight`, `Scene` và `Caption`.
3. **Client-side Filtering & Fuzzy Match (Fuse.js)**:
   - Áp dụng ý tưởng từ SnapSeek 3.0: Thay vì gửi yêu cầu filter OCR/Scene liên tục lên backend, backend chỉ cần trả về Top-N (vd: 500) kết quả từ Vector Search.
   - Trình duyệt sẽ dùng thư viện `Fuse.js` để tìm kiếm mờ (fuzzy match) theo thời gian thực (real-time) dựa trên metadata (OCR, Scene, Caption) đi kèm các kết quả đó.
   - Giúp giảm tải Server GPU tối đa và tốc độ phản hồi UI cực nhanh (instant feedback).

## 3. Lộ trình code tiếp theo
1. Dùng CSS/JS hiện đại (hoặc Bootstrap/Tailwind qua CDN) để cải tạo layout `index.html`.
2. Bổ sung CDN thư viện `Fuse.js` vào file HTML.
3. Tạo Component UI cho `Metadata Inspector` và `Client-side Filters` (Thanh search mờ).
4. Giữ lại toàn bộ code `script` (hàm `searchText()`, `uploadAndSearch()`, `runTrake()`, `makeSubmission()`, v.v.) và chỉ cập nhật cách lấy data từ DOM HTML mới.
5. Cập nhật API backend (nếu cần) để đảm bảo response trả về có kèm đầy đủ dữ liệu OCR/Scene phục vụ Fuse.js ở UI.
