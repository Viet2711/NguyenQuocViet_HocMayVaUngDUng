### 1. [Công Nghệ Sử Dụng]

**Numpy**: Đây là một thư viện quan trọng cho khoa học dữ liệu và tính toán số học. Nó cung cấp các phép toán mạnh mẽ trên mảng (arrays), giúp thực hiện các thao tác số học với hiệu suất cao. Numpy đóng vai trò cốt lõi trong việc xử lý dữ liệu số và tối ưu hóa các phép tính ma trận.

**Pandas**: Một thư viện mạnh mẽ giúp quản lý và phân tích dữ liệu với các thao tác trực quan. Pandas có khả năng đọc và ghi dữ liệu từ nhiều định dạng khác nhau (như .csv, .xls, .xlsx). Điều này giúp nó trở thành công cụ đắc lực trong việc xử lý và phân tích dữ liệu trong khoa học dữ liệu.

**Scikit-learn (sklearn)**: Là thư viện cung cấp các thuật toán học máy phổ biến, cùng với các công cụ hỗ trợ cho việc chia dữ liệu (train-test split) và đánh giá mô hình. Trong đoạn mã này, `train_test_split` của sklearn được dùng để chia tập dữ liệu thành tập huấn luyện (train) và kiểm tra (test), nhằm chuẩn bị cho quá trình huấn luyện và đánh giá mô hình.

### 2. [Thuật Toán]

- **[Tải dữ liệu]**: Hàm `loadExcel` trả về một DataFrame Pandas chứa dữ liệu từ file Excel.
  
- **[Huấn luyện và kiểm tra]**: Hàm `splitTrainTest` thực hiện việc xáo trộn ngẫu nhiên các chỉ mục của dữ liệu và chia dữ liệu thành hai phần: một phần dùng để huấn luyện (train) và một phần để kiểm tra (test). Tập huấn luyện được sử dụng để xây dựng mô hình, trong khi tập kiểm tra dùng để đánh giá hiệu suất mô hình.

- **[Chia tập dữ liệu]**: Hàm `splitTrainTest` chia dữ liệu dựa trên biến target (cột mà chúng ta muốn dự đoán). Tham số `ratio` được sử dụng để xác định tỉ lệ của dữ liệu dành cho tập kiểm tra.

- **[Tính trung bình lớp]**: Hàm `mean_class` sử dụng phương pháp nhóm (groupby) để tính trung bình của các đặc trưng (features) trong mỗi lớp của biến mục tiêu. Điều này có nghĩa là đối với mỗi lớp trong tập dữ liệu (ví dụ như các loài hoa trong tập dữ liệu Iris), hàm sẽ tính toán trung bình các đặc trưng.

- **[Dự đoán bằng khoảng cách Euclid]**: Hàm `target_pred` sử dụng khoảng cách Euclid để so sánh các điểm dữ liệu trong tập kiểm tra với các giá trị trung bình của từng lớp đã tính toán. Khoảng cách Euclid là một phương pháp đo độ tương đồng giữa hai điểm trong không gian nhiều chiều.

- **[Kết quả]**: Kết quả dự đoán sau đó được so sánh với các giá trị thực tế trong tập kiểm tra để xác định độ chính xác của mô hình.

### 3. [Hiển Thị Kết Quả]

- **Kết quả**:
  Khi chạy thử code `centroid_practice`:
  ```python
  data = loadExcel('Iris.xls')
  data_train, X_test, y_test = splitTrainTest(data, 'iris', ratio=0.3)
  ```

- **In kết quả**:
  - Dữ liệu huấn luyện: `print(data_train)`
  - Dữ liệu kiểm tra (test features): `print(X_test)`
  - Nhãn thực tế của tập kiểm tra: `print(y_test)`

Hàm bao gồm các bước:
- Tải dữ liệu Iris và chia thành tập huấn luyện và tập kiểm tra.
- Tính toán giá trị trung bình cho mỗi lớp trong tập huấn luyện.
- Dự đoán lớp cho các mẫu trong tập kiểm tra dựa trên khoảng cách Euclid.
BÀI TẬP 2: KNN_BT2