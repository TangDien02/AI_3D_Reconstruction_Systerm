import os
import sys
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# Khởi tạo Flask và chỉ định thư mục chứa file index.html
app = Flask(__name__, template_folder='templates')
CORS(app)

# Thêm thư mục hiện tại vào hệ thống để có thể import các file từ 'src' nếu cần
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 1. Route hiển thị giao diện chính
@app.route('/')
def home():
    return render_template('index.html')

# 2. Route nhận ảnh từ index.html và xử lý tạo 3D
@app.route('/generate', methods=['POST'])
def generate_3d():
    try:
        # Kiểm tra xem giao diện có gửi file ảnh lên không
        if 'image' not in request.files:
            return jsonify({'error': 'Không tìm thấy file ảnh gửi lên từ giao diện'}), 400
            
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'File ảnh không hợp lệ'}), 400

        # Tạo một thư mục tạm để lưu ảnh đầu vào nếu cần
        upload_dir = os.path.join(os.path.dirname(__file__), 'data', 'tmp_uploads')
        os.makedirs(upload_dir, exist_ok=True)
        
        input_image_path = os.path.join(upload_dir, file.filename)
        file.save(input_image_path)

        # ---------------------------------------------------------------------
        # TÍCH HỢP MODEL CỦA BẠN TẠI ĐÂY:
        # Bạn hãy import hàm chạy mô hình từ 'src' hoặc 'main_workflow.py' vào.
        # Ví dụ giả định:
        # from src.inference import reconstruct_3d_from_image
        # obj_file_path = reconstruct_3d_from_image(image_path=input_image_path)
        #
        # Với đầu ra mong muốn là nội dung chuỗi (text) của file .obj để trả về.
        # ---------------------------------------------------------------------
        
        # TẠM THỜI: Đây là đoạn giả lập (Mock dữ liệu) để bạn test giao diện trước
        # Sau khi bạn tích hợp model thật, hãy đọc nội dung file .obj của bạn và gán vào biến `obj_text`
        import time
        time.sleep(2) # Giả lập model xử lý trong 2 giây
        
        obj_text = """
        v -0.5 -0.5 0.5
        v 0.5 -0.5 0.5
        v 0.5 0.5 0.5
        v -0.5 0.5 0.5
        v -0.5 -0.5 -0.5
        v 0.5 -0.5 -0.5
        v 0.5 0.5 -0.5
        v -0.5 0.5 -0.5
        f 1 2 3 4
        f 5 6 7 8
        f 1 2 6 5
        f 2 3 7 6
        f 3 4 8 7
        f 4 1 5 8
        """
        # ---------------------------------------------------------------------

        # Trả về chuỗi dữ liệu OBJ trực tiếp cho Three.js ở giao diện hiển thị
        return obj_text, 200, {'Content-Type': 'text/plain; charset=utf-8'}

    except Exception as e:
        return jsonify({'error': f'Lỗi xảy ra trong quá trình xử lý: {str(e)}'}), 500

if __name__ == '__main__':
    # Chạy chạy local tại cổng 5000 trùng khớp với file index.html của bạn
    app.run(host='127.0.0.1', port=5000, debug=True)