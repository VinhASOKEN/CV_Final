Project này bao gồm source code implement cho ba mô hình khác nhau: Dino, Resnet50 và ViT. Mỗi mô hình có một thư mục riêng, và các tệp trọng số cho các mô hình này đã được public trong từng thư mục tương ứng.

Trước khi chạy code infer, hãy đảm bảo bạn cập nhật đường dẫn đến các tệp trọng số trong tệp cấu hình (config.py) cho mỗi mô hình. Ngoài ra, chỉ định đường dẫn đến thư mục bạn muốn kiểm thử trong cùng tệp cấu hình.

Có hai file infer chính được cung cấp:
1. zinfer_all.py

File này được thiết kế để đánh giá độ chính xác trên một tập test với cấu trúc thư mục trong đó mỗi thư mục con tương ứng với một class khác nhau và chứa các ảnh cần test của class đó. Hãy điều chỉnh config trước khi chạy.

2. zinfer_single.py

File này cho phép bạn đánh giá các class dự đoán cho mỗi ảnh trong một thư mục test. Hãy cập nhật config với các đường dẫn phù hợp trước khi thực thi.
