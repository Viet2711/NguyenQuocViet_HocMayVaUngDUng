## Dự đoán Loại Thuốc với Học Máy

Bài viết này trình bày cách xây dựng một mô hình học máy đơn giản để dự đoán loại thuốc phù hợp cho bệnh nhân dựa trên các thông tin sức khỏe của họ.

**Công nghệ:**

Chúng ta sử dụng các thư viện Python phổ biến trong lĩnh vực khoa học dữ liệu:

* **NumPy:** Xử lý dữ liệu số và ma trận hiệu quả.
* **Pandas:** Đọc dữ liệu từ file CSV, tổ chức và phân tích dữ liệu dễ dàng.
* **Scikit-learn:** Cung cấp các thuật toán học máy, chia dữ liệu và đánh giá mô hình.

**Thuật toán:**

1. **Nạp dữ liệu:** Sử dụng Pandas để đọc dữ liệu bệnh nhân từ file CSV. Dữ liệu bao gồm các đặc trưng (tuổi, giới tính, huyết áp, cholesterol...) và loại thuốc tương ứng.

2. **Chuẩn bị dữ liệu:** Chia dữ liệu thành hai tập: 
    * **Tập huấn luyện:** Dùng để "dạy" cho mô hình.
    * **Tập kiểm tra:** Dùng để kiểm tra độ chính xác của mô hình sau khi huấn luyện.

3. **Xây dựng mô hình:**  Sử dụng hai thuật toán:
    * **Cây quyết định:**  Xây dựng một "cây" các quy tắc quyết định để phân loại. Ví dụ: Nếu tuổi > 50 và huyết áp cao thì dùng thuốc A.
    * **Rừng ngẫu nhiên:** Kết hợp nhiều cây quyết định để tăng độ chính xác và ổn định.

4. **Huấn luyện mô hình:** Sử dụng tập huấn luyện để điều chỉnh mô hình sao cho dự đoán chính xác nhất có thể.

5. **Đánh giá:** Sử dụng tập kiểm tra để đánh giá hiệu suất của mô hình. Kết quả đánh giá (độ chính xác, ma trận nhầm lẫn...) được hiển thị trong hình ảnh.

**Hình ảnh kết quả:**

     Age  Sex  BP  Cholesterol  Na_to_K   Drug
0     23    1   2            1   25.355  DrugY
1     47    0   0            1   13.093  drugC
2     47    0   0            1   10.114  drugC
3     28    1   1            1    7.798  drugX
4     61    1   0            1   18.043  DrugY
..   ...  ...  ..          ...      ...    ...
195   56    1   0            1   11.567  drugC
196   16    0   0            1   12.006  drugC
197   52    0   1            1    9.894  drugX
198   23    0   1            0   14.020  drugX
199   40    1   0            0   11.349  drugX

[200 rows x 6 columns]
0      4
1      2
2      2
3      3
4      4
      ..
195    2
196    2
197    3
198    3
199    3
Name: Drug, Length: 200, dtype: int64

**Kết luận:**

Mô hình học máy có thể giúp dự đoán loại thuốc phù hợp cho bệnh nhân dựa trên thông tin sức khỏe. Cây quyết định và Rừng ngẫu nhiên là hai thuật toán hiệu quả cho bài toán này.


**Ghi chú:**

* Việc tinh chỉnh các tham số của mô hình (ví dụ: độ sâu của cây quyết định) là rất quan trọng để đạt được hiệu quả tốt nhất.
* Rừng ngẫu nhiên thường cho kết quả chính xác hơn cây quyết định đơn lẻ.


