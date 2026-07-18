# ĐẶC TẢ KỸ THUẬT: HỆ THỐNG KHỐI TÌM KIẾM ĐỘC LẬP (INDEPENDENT QUERY BLOCKS)

Tài liệu này đặc tả thiết kế giao diện phân chia các chức năng tìm kiếm thành các khối độc lập nhằm giải quyết triệt để lỗi ghi đè kết quả giữa tìm kiếm Text và tìm kiếm Ảnh.

---

## 1. Nguyên lý Thiết kế (Core Concept)

Hệ thống giao diện tìm kiếm sẽ được chia thành các **Khối chức năng độc lập (Independent Blocks)**. Mỗi khối là một đơn vị khép kín chứa cả phần nhập liệu và phần hiển thị kết quả riêng:

*   **Khối 1: Tìm kiếm bằng Văn bản (Text Search Block)**
*   **Khối 2: Tìm kiếm bằng Hình ảnh (Image Search Block)**
*   **Khối 3: Tìm kiếm Chuỗi thời gian (Temporal Search Block)**
*   **Khối 4: Tìm kiếm OCR (OCR Search Block)**

### Quy tắc hoạt động:
1.  **Giao diện toàn vẹn**: Khi người dùng click chọn một Khối (ví dụ: Khối Text), màn hình sẽ chuyển đổi toàn bộ sang giao diện của khối đó (gồm ô nhập liệu Text + Lưới kết quả của Text).
2.  **Độc lập hoàn toàn (Sandboxed)**: Việc chạy truy vấn ở Khối này không được ảnh hưởng, thay đổi hay ghi đè lên kết quả của Khối khác.
3.  **Bảo toàn trạng thái**: Khi người dùng chuyển từ Khối Text sang Khối Ảnh để làm việc, rồi sau đó click quay lại Khối Text, toàn bộ câu lệnh đã gõ và các kết quả tìm kiếm trước đó của Khối Text vẫn phải giữ nguyên như cũ (không bị reset hay chạy lại), trừ khi người dùng chủ động nhấn nút xóa kết quả.

---

## 2. Đặc tả Chi tiết cho Lập trình viên (Dev Specifications)

### A. Sơ đồ Cấu trúc Giao diện (UI Layout)

Màn hình chính được thiết kế chia làm hai phần: Thanh chọn Khối (Block Selector) ở trên và Vùng hiển thị Khối hoạt động ở dưới.


### B. Quản lý Trạng thái phía Client (Client-side State Management)

Để đảm bảo kết quả không bị mất khi chuyển khối, lập trình viên cần lưu trữ trạng thái của từng khối riêng biệt trong bộ nhớ (Client memory) và không được giải phóng bộ nhớ này khi chuyển khối.

#### Cấu trúc dữ liệu State đề xuất (Javascript/React/Vue):

```javascript
// State tổng quản lý toàn bộ giao diện tìm kiếm
const searchWorkspaceState = {
  // Xác định khối nào đang được hiển thị trên màn hình
  activeBlock: "text_search", // "text_search" hoặc "image_search" hoặc "temporal_search"
  
  // Trạng thái lưu trữ của Khối Text Search
  textBlock: {
    inputQuery: "a red car",      // Lưu câu lệnh đang nhập dở
    results: [ /* Danh sách ảnh kết quả trả về từ API Text Search */ ],
    scrollPosition: 0             // Lưu vị trí cuộn trang của lưới kết quả Text
  },

  // Trạng thái lưu trữ của Khối Image Search
  imageBlock: {
    selectedImageFile: null,      // File ảnh người dùng upload làm truy vấn
    results: [ /* Danh sách ảnh kết quả từ API Image Search */ ],
    scrollPosition: 0             // Lưu vị trí cuộn trang của lưới kết quả Ảnh
  },

  // Trạng thái lưu trữ của Khối Temporal Search (TRAKE)
  temporalBlock: {
    events: ["man walking", "red car"],
    results: [ /* Danh sách kết quả chuỗi sự kiện */ ],
    scrollPosition: 0
  }
};
```

---

### C. Luồng xử lý Sự kiện (Event Flow)

1.  **Người dùng click chuyển Khối (Switch Block)**:
    *   *Bước 1*: Lưu lại vị trí cuộn trang (`scrollTop` của lưới kết quả) vào trạng thái của Khối hiện tại (ví dụ: lưu vào `textBlock.scrollPosition`).
    *   *Bước 2*: Thay đổi biến `activeBlock` sang tên Khối mới (ví dụ: `image_search`).
    *   *Bước 3*: Giao diện tự động ẩn Khối cũ và hiển thị Khối mới.
    *   *Bước 4*: Khôi phục lại vị trí cuộn trang của Khối mới (đọc từ `imageBlock.scrollPosition`).
    *   *Lưu ý quan trọng*: Quá trình chuyển đổi này chỉ thay đổi hiển thị ở Frontend, **tuyệt đối không gửi yêu cầu API mới lên Server**.

2.  **Người dùng chạy Truy vấn (Run Query)**:
    *   *Bước 1*: Gửi dữ liệu đầu vào của Khối hiện tại lên server qua API tương ứng (Text -> `/api/search_text`, Image -> `/api/search_image`).
    *   *Bước 2*: Nhận kết quả từ server và chỉ cập nhật vào mảng kết quả của Khối đó (ví dụ: `textBlock.results`).
    *   *Bước 3*: Đặt vị trí cuộn trang của Khối đó về 0.

3.  **Người dùng bấm nút Xóa kết quả (Clear Results)**:
    *   *Hành động*: Trên mỗi khối sẽ có một nút bấm "Xóa kết quả" riêng.
    *   *Bước 1*: Reset mảng `results` của Khối hiện tại về rỗng (`[]`).
    *   *Bước 2*: Reset phần nhập liệu (Input) của Khối đó về trạng thái mặc định.

---

## 3. Hướng dẫn lập trình giao diện giữ nguyên DOM (DOM Retention Guide)

Để tránh việc lưới ảnh bị tải lại hoặc mất trạng thái cuộn trang khi ẩn hiện các Khối, lập trình viên nên sử dụng kỹ thuật giữ nguyên DOM (không hủy và tạo lại component) bằng cách ẩn hiện thông qua CSS:

```html
<!-- Mẫu cấu trúc DOM giữ nguyên trạng thái kết quả của các khối -->
<div class="workspace-container">
  
  <!-- KHỐI 1: TEXT SEARCH (Chỉ hiển thị khi activeBlock === 'text_search') -->
  <div class="query-block" id="text-search-block" style="display: activeBlock === 'text_search' ? 'block' : 'none'">
    <div class="input-area">
      <!-- Input của riêng khối Text -->
      <input type="text" value={textBlock.inputQuery} />
      <button onclick="runTextSearch()">Chạy truy vấn</button>
      <button onclick="clearTextBlock()">Xóa kết quả</button>
    </div>
    <div class="results-grid">
      <!-- Grid hiển thị kết quả của riêng khối Text -->
      <!-- Dữ liệu được render từ textBlock.results -->
    </div>
  </div>

  <!-- KHỐI 2: IMAGE SEARCH (Chỉ hiển thị khi activeBlock === 'image_search') -->
  <div class="query-block" id="image-search-block" style="display: activeBlock === 'image_search' ? 'block' : 'none'">
    <div class="input-area">
      <!-- Input của riêng khối Ảnh -->
      <input type="file" />
      <button onclick="runImageSearch()">Chạy truy vấn</button>
      <button onclick="clearImageBlock()">Xóa kết quả</button>
    </div>
    <div class="results-grid">
      <!-- Grid hiển thị kết quả của riêng khối Ảnh -->
      <!-- Dữ liệu được render từ imageBlock.results -->
    </div>
  </div>

</div>
```
