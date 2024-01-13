Dự án này bao gồm mã nguồn triển khai cho ba mô hình khác nhau: Dino, Resnet50 và ViT. Mỗi mô hình có một thư mục riêng, và các tệp trọng số cho các mô hình này đã được công khai trong từng thư mục tương ứng.

Bắt Đầu
Trước khi chạy mã nguồn dự đoán, hãy đảm bảo bạn cập nhật đường dẫn đến các tệp trọng số trong tệp cấu hình (config.py) cho mỗi mô hình. Ngoài ra, chỉ định đường dẫn đến thư mục bạn muốn kiểm thử trong cùng tệp cấu hình.

Có hai tệp mã nguồn chính được cung cấp:

1. zinfer_all.py
Tệp mã nguồn này được thiết kế để đánh giá độ chính xác trên một tập thử nghiệm với cấu trúc thư mục trong đó mỗi thư mục con tương ứng với một lớp khác nhau. Hãy điều chỉnh tệp cấu hình trước khi chạy.

2. zinfer_single.py
Tệp mã nguồn này cho phép bạn đánh giá các nhãn dự đoán cho mỗi ảnh trong một thư mục kiểm thử. Cập nhật tệp cấu hình với các đường dẫn phù hợp trước khi thực thi.

Đảm bảo rằng các đường dẫn đến các tệp trọng số và thư mục kiểm thử đã được thiết lập đúng trong tệp cấu hình (config.py) cho mỗi mô hình.
Các tệp trọng số cho mỗi mô hình có sẵn công khai trong thư mục tương ứng của chúng.
