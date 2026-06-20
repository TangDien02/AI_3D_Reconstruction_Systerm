# 📝 Nhật ký Cải tiến Kiến trúc & Kỹ thuật (Changelog for Report)

Tài liệu này tổng hợp các thay đổi mang tính bước ngoặt trong quá trình tinh chỉnh mô hình AI 3D Reconstruction.

### 1. Tích hợp định dạng `.ply` cho Pipeline Suy luận (Inference)
*   **Vấn đề:** Ban đầu, pipeline chỉ xuất kết quả dạng ma trận `.npy`. Định dạng này khó có thể đánh giá trực quan (Qualitative Evaluation) thông qua các phần mềm đồ họa 3D chuyên dụng.
*   **Giải pháp:** Cập nhật script `compare_pointclouds.py` và `baseline_inference.py`. Sử dụng thư viện I/O để tự động lưu song song định dạng Polygon File Format (`.ply`) bên cạnh `.npy`.
*   **Ý nghĩa báo cáo:** Cho phép import trực tiếp Point Cloud dự đoán và Ground Truth vào phần mềm Blender / MeshLab để render chất lượng cao, phục vụ cho việc soi chi tiết bề mặt và tạo hình ảnh minh họa (figures) trực quan cho báo cáo.

### 2. Bật & Phân tích các Hàm Mất mát Nâng cao (Advanced Geometric Regularizers)
*   **Vấn đề:** Khi chỉ sử dụng Chamfer Distance (CD) đơn thuần, mạng MLP có xu hướng "ăn gian" bằng cách vón cục điểm (clumping) tại các vùng trung tâm để tối thiểu hóa khoảng cách trung bình, dẫn đến lưới điểm lộn xộn và bỏ sót các chi tiết mảnh (ví dụ: chân ghế).
*   **Giải pháp:** Đưa vào kết hợp 3 hàm loss phụ học theo cơ chế Coarse-to-Fine:
    *   **Repulsion Loss:** Phạt các điểm dự đoán nằm quá sát nhau, ép chúng phải tản ra.
    *   **Uniformity Loss:** Ép sự phân bố khoảng cách giữa các điểm lân cận phải đồng đều, tạo ra cấu trúc dạng lưới (grid-like) mượt mà.
    *   **Detail Coverage Loss:** Ép mạng phải phủ điểm lên những vùng Ground Truth khó tiếp cận.
*   **Ý nghĩa báo cáo:** Giải thích được hiện tượng *Trade-off (Đánh đổi)*: Việc bật Loss nâng cao làm tăng nhẹ chỉ số Chamfer Distance (do điểm bị ép dàn đều thay vì bám dính cục bộ), nhưng đổi lại cải thiện đột phá chỉ số **Density Score** (lên tới 0.85) và giảm triệt để **Clump Ratio** (xuống dưới 5%), mang lại kết quả thị giác (Visual Completeness) vượt trội.

### 3. Sửa lỗi Bẫy Khởi tạo của PEFT Adapter (Zero-Initialization Bug)
*   **Vấn đề:** Khi tích hợp Adapter (dạng Residual) vào sau ResNet50 để fine-tune, mô hình bị kẹt ở Local Optimum (Chamfer Distance plateau ở mốc `0.0103`, không thể giảm sâu xuống `0.0094` như mô hình không có Adapter). Nguyên nhân là do lớp Linear cuối cùng của Adapter sử dụng Kaiming Initialization mặc định, tạo ra nhiễu (noise) khổng lồ cộng thẳng vào feature của ResNet50 ngay từ epoch 0. Do encoder bị đóng băng (freeze) ở 6 epoch đầu, decoder bị ép phải hội tụ trên một nền feature nhiễu.
*   **Giải pháp:** Cải tiến module `AdapterBlock` bằng cách áp dụng **Zero-Initialization** cho trọng số (weight) và bias của lớp Up-projection.
*   **Ý nghĩa báo cáo:** Đảm bảo Adapter hoạt động như một hàm đồng nhất (Identity Function) tại thời điểm khởi tạo (`features + 0 = features`). Khám phá này đã giúp mô hình thoát khỏi điểm nghẽn, hội tụ thành công về mức CD `0.0094` (với feature sạch), chứng minh kiến trúc PEFT không hề gây cản trở dòng thông tin nếu được thiết kế chuẩn mực.

### 4. Khắc phục Hiệu ứng Co rút Điểm (Point Shrinkage Effect) tại Vùng Biên
*   **Vấn đề:** Khi render kết quả, point cloud dự đoán luôn bị "co lại" và không bao giờ chạm tới được các đường viền/góc cạnh ngoài cùng của Ground Truth.
*   **Nguyên nhân:** Có sự cộng hưởng từ 2 yếu tố:
    1. Hàm kích hoạt đầu ra `nn.Tanh()` bị bão hòa (Vanishing Gradient) khi tọa độ tiến gần đến biên `±0.7` đến `±1.0`. Gradient gần bằng 0 khiến mô hình bất lực trong việc đẩy điểm ra xa hơn.
    2. Hàm Chamfer Distance thiên vị vùng trung tâm, khiến mô hình thà co cụm ở giữa để an toàn còn hơn mạo hiểm đẩy điểm ra biên (nơi dễ bị phạt nặng nếu sai lệch).
*   **Giải pháp:**
    *   Thay thế toàn bộ `nn.Tanh()` bằng **`nn.Hardtanh(-1.0, 1.0)`**: Giữ nguyên gradient = 1.0 trong toàn bộ không gian hợp lệ, cho phép mô hình thoải mái đẩy điểm ra viền vật thể.
    *   Tăng tham số **`chamfer_gt_weight` lên 1.5**: Tăng hình phạt đối với việc bỏ sót viền, ép mô hình phải nội suy các điểm bao phủ toàn bộ giới hạn của Ground Truth.
*   **Ý nghĩa báo cáo:** Là minh chứng tuyệt vời cho thấy việc thấu hiểu sự tương tác giữa **Activation Function** và **Loss Landscape** trong không gian 3D có thể giải quyết các khiếm khuyết nghiêm trọng về mặt hình học mà các công cụ debug thông thường không thể nhận ra.
