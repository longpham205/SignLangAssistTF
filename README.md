=== Cài đặt ban đầu ===

Mở termial

//Trỏ tới thư mục dự án

cd <Nơi lưu dự án>\SignLangAssistTF

//Tạo môi trường ảo

python -m venv venv

// Chạy môi trường ảo

.\venv\Scripts\Activate.ps1 

//Tải thư viện 

pip install -r requirements.txt

=== Hướng chẫn chạy mô hình ===

1. Tạo data
    - Chạy file make_data.py để tự tạo các dữ liệu mong muốn:   
        python sign_to_text\dataset\make_data.py
    - Chạy flie split_data.py để chia dữ liệu train và val:     
        python sign_to_text\dataset\make_data.py
    - Có thể tải nhanh data tham khảo rồi lưu vào dataset\ :


2. Train mô hình
    2.1. Sign To Text
        - Chạy file train_chars.py để huấn luyện images kí tự tĩnh:     
            python sign_to_text\src\main\train_chars.py
        - Chạy file train_chars.py để huấn luyện video từ động:         
            python sign_to_text\src\main\train_videos.py
        - Có thể tải kết quả train tham khảo rồi lưu vào checkponit\ :

3. Test mô hình
    3.1. Sign To Text
        - Chạy file infer.py để test cả mô hình: 
            python sign_to_text\src\main\infer.py
        - Chạy file infer.py để test nhận diện images kí tự tĩnh: 
            python sign_to_text\src\main\infer_char.py
        - Chạy file infer.py để test nhận diện video từ động : 
            python sign_to_text\src\main\infer.py

