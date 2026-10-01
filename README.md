# CO3117 - Học máy (Machine Learning): Xây dựng mô hình từ con số không

Dự án này là tập hợp các thuật toán Học máy được tự xây dựng từ đầu (from scratch) bằng Python, phục vụ cho việc học tập và nghiên cứu môn học **CO3117 - Học máy** (HCMUT).

Mục tiêu cốt lõi không phải là tạo ra một thư viện nhanh nhất, mà là hiểu sâu bản chất toán học, cách biểu diễn dữ liệu và luồng hoạt động của các thuật toán kinh điển. Tất cả các mô hình tự code đều được đặt lên bàn cân so sánh với `scikit-learn` về độ chính xác và thời gian thực thi.

## Cấu trúc dự án

```text
ML-From-Scratch/
├── data/                       # Các tập dữ liệu
│   └── play_tennis.csv         # Tập dữ liệu mẫu
├── src/
│   ├── models/                 # Chứa các thuật toán tự triển khai
│   │   ├── base_model.py       # Lớp trừu tượng định nghĩa .fit() và .predict()
│   │   └── decision_tree.py    # Thuật toán Cây quyết định (Decision Tree)
│   ├── preprocessing/          # Module xử lý dữ liệu (Label Encoding, Scaling)
│   │   └── encoders.py         # Chứa các lớp mã hóa nhãn và tính năng
│   ├── utils/                  # Các hàm tiện ích (Metrics, Data Loader)
│   │   ├── data_loader.py      # Script hỗ trợ đọc và tải dữ liệu
│   │   ├── logger.py           # Bộ theo dõi và ghi log quá trình huấn luyện
│   │   └── metrics.py          # Các hàm đánh giá hiệu suất mô hình
│   └── main.py                 # Kịch bản điều phối và khởi chạy dự án
├── requirements.txt
└── README.md
```

## ⚙️ Cài đặt môi trường

Khuyến nghị sử dụng môi trường ảo (Virtual Environment) để tránh xung đột thư viện:

1. Tạo môi trường ảo:

   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

2. Cài đặt các thư viện phụ thuộc:
   ```bash
   pip install -r requirements.txt
   ```

## Hướng dẫn sử dụng

Bạn có thể chạy các mô hình học máy trực tiếp từ terminal thông qua file `main.py`. Dự án sử dụng giao diện dòng lệnh (CLI) để quản lý các mô hình, tập dữ liệu và các chế độ thực thi khác nhau.

### Khởi chạy cơ bản

Để huấn luyện và đánh giá một mô hình trên một tập dữ liệu cụ thể, hãy sử dụng câu lệnh sau:

```bash
python src/main.py --model MODEL-NAME --data data/DATA-FILE-NAME
```

### Các tham số dòng lệnh (Flags)

| Cờ (Flag)   | Loại         | Giải thích                                                                                                                                                                                                                        |
| :---------- | :----------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--model`   | **Bắt buộc** | Tên của mô hình bạn muốn khởi tạo và chạy (ví dụ: `decision_tree`).                                                                                                                                                               |
| `--data`    | **Bắt buộc** | Đường dẫn tương đối đến file dữ liệu của bạn (ví dụ: `data/play_tennis.csv`).                                                                                                                                                     |
| `--compare` | _Tùy chọn_   | Kích hoạt chế độ so sánh song song. Chế độ này sẽ huấn luyện một mô hình Scikit-Learn tương đương trên cùng tập dữ liệu và in ra bảng so sánh các chỉ số (Accuracy, Precision, Recall, F1-Score).                                 |
| `--debug`   | _Tùy chọn_   | Kích hoạt `TrainingTracker`. Chế độ này sẽ lưu lại chi tiết từng bước tính toán toán học (như Parent Entropy, Thresholds, và Information Gain) vào một file riêng tại `logs/training_log.txt` mà không làm rối màn hình terminal. |

### Khởi chạy chế độ nâng cao

```bash
python src/main.py --model MODEL-NAME --data data/DATA-FILE-NAME --compare --debug
```

## Tiến độ triển khai (Roadmap)

### Dự án sẽ lần lượt xây dựng các mô hình sau:

- **Chương 2:** Cây quyết định (Decision Tree - ID3/C4.5)
- **Chương 3:** Mạng nơron nhân tạo (Perceptron & Backpropagation)
- **Chương 4:** Naive Bayes
- **Chương 5:** Giải thuật di truyền (Genetic Algorithm)
- **Chương 6:** Các mô hình đồ thị (Graphical Models - Bayes network & HMM)
- **Chương 7:** Máy vectơ hỗ trợ (SVM)
- **Chương 8:** Thu giảm số chiều (Dimensionality Reduction - PCA & LDA)
- **Chương 9:** Học hợp quần (Bagging & Boosting)
- **Chương 10:** Hồi quy Logistic (Logistic Regression)

## Bộ dữ liệu kiểm thử

- **Play Tennis:** Tập dữ liệu phân loại nhỏ (14 mẫu) dùng để debug logic tính toán Toán học (Entropy, Information Gain) và đối chiếu thủ công.

```

```
