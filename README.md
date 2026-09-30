# CO3117 - Học máy (Machine Learning): Xây dựng mô hình từ con số không

Dự án này là tập hợp các thuật toán Học máy được tự xây dựng từ đầu (from scratch) bằng Python, phục vụ cho việc học tập và nghiên cứu môn học **CO3117 - Học máy** (HCMUT).

Mục tiêu cốt lõi không phải là tạo ra một thư viện nhanh nhất, mà là hiểu sâu bản chất toán học, cách biểu diễn dữ liệu và luồng hoạt động của các thuật toán kinh điển. Tất cả các mô hình tự code đều được đặt lên bàn cân so sánh với `scikit-learn` về độ chính xác và thời gian thực thi.

## Cấu trúc dự án

```text
ML-From_Scratch/
├── data/                       # Các tập dữ liệu
├── src/
│   ├── models/                 # Chứa các thuật toán tự triển khai
│   │   ├── base_model.py       # Lớp trừu tượng định nghĩa .fit() và .predict()
│   │   └── ...
│   ├── preprocessing/          # Module xử lý dữ liệu (Label Encoding, Scaling)
│   ├── utils/                  # Các hàm tiện ích (Metrics, Data Loader)
│   └── main.py                 # Kịch bản điều phối và so sánh các mô hình
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

Chạy kịch bản chính để kiểm thử và so sánh một mô hình cụ thể. (Phần này sẽ được cập nhật khi hoàn thiện tính năng truyền tham số CLI).

Ví dụ dự kiến:

```bash
python src/main.py --model decision_tree --data data/play_tennis.csv
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
