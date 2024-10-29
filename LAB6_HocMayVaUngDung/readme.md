# Mô Hình Phân Loại Dữ Liệu Fashion MNIST Sử Dụng MLP

## 1. Công Nghệ Sử Dụng
Bài lab này sử dụng các công nghệ và thư viện sau:
- **Ngôn ngữ lập trình**: Python
- **Thư viện học sâu**: PyTorch để xây dựng và huấn luyện mô hình
- **Thư viện hỗ trợ**: 
  - **Torchvision**: để tải và xử lý bộ dữ liệu FashionMNIST
  - **NumPy**: để tính toán các hàm mất mát (loss functions) và các phép tính đại số
  - **Matplotlib**: để trực quan hóa kết quả

## 2. Thuật Toán Machine Learning Sử Dụng
Bài này sử dụng thuật toán **Multi-Layer Perceptron (MLP)**, một dạng mạng nơ-ron đơn giản, để thực hiện phân loại hình ảnh từ bộ dữ liệu FashionMNIST. 

Cách hoạt động của MLP trong bài toán phân loại:
1. **Flattening**: Ảnh 28x28 được chuyển thành một vector có kích thước 784 để phù hợp với đầu vào của mô hình.
2. **Hidden Layer**: Một lớp ẩn với 256 nút và hàm kích hoạt ReLU được sử dụng để xử lý đầu vào.
3. **Output Layer**: Lớp đầu ra có 10 nút, mỗi nút đại diện cho một lớp trong bộ dữ liệu FashionMNIST.
4. **Loss Function**: Hàm mất mát CrossEntropy được sử dụng để tính độ lệch giữa dự đoán của mô hình và nhãn thực tế.
5. **Optimization**: Thuật toán **SGD (Stochastic Gradient Descent)** giúp tối ưu hóa trọng số mô hình qua từng bước huấn luyện.

## 3. Hiển Thị Kết Quả
Kết quả của mô hình được hiển thị thông qua biểu đồ mô tả **Loss** và **Accuracy** sau mỗi epoch. Kết quả dự đoán cuối cùng của mô hình sẽ hiển thị các nhãn dự đoán cho hình ảnh của người dùng tải lên.

## 4. Đánh Giá Giữa Các Thuật Toán
Hiện tại, bài toán này chỉ sử dụng MLP, nhưng có thể mở rộng đánh giá bằng cách sử dụng các mô hình khác như **Convolutional Neural Network (CNN)**. CNN thường cho kết quả tốt hơn trên bài toán phân loại ảnh nhờ khả năng tự động trích xuất đặc trưng.

## Cách Sử Dụng

### 1. Tải Bộ Dữ Liệu
Code sử dụng bộ dữ liệu **FashionMNIST**, bao gồm 60,000 ảnh train và 10,000 ảnh test. Bộ dữ liệu này được tự động tải về từ thư viện `torchvision`.

### 2. Khởi Tạo và Huấn Luyện Mô Hình
Mô hình MLP được khởi tạo với các tham số mặc định và được huấn luyện trên 10 epoch.

### 3. Lưu Trọng Số
Sau khi huấn luyện, mô hình sẽ được lưu dưới dạng file `.pth` để phục vụ việc sử dụng lại mà không cần huấn luyện lại từ đầu.
```python
torch.save(model, "MLP_dress.pth")
