# Open Food Facts Dataset Analysis

## Giới thiệu

**Open Food Facts** là một bộ dữ liệu mở, được cộng đồng đóng góp, cung cấp thông tin chi tiết về các sản phẩm thực phẩm trên toàn thế giới.
Repository này tập trung vào việc **khai thác, phân tích và xây dựng mô hình Machine Learning** dựa trên dữ liệu Open Food Facts.

---

## 🎯 Mục tiêu dự án

* Tiền xử lý và làm sạch dữ liệu Open Food Facts
* Xây dựng mô hình **phân loại (Classification)** sản phẩm thực phẩm
* Thực hiện **phân cụm (Clustering)** để khám phá các nhóm sản phẩm tương đồng
* Sử dụng **Luật kết hợp** để tìm hiểu mối quan hệ giữa các thành phần dinh dưỡng và nhãn thực phẩm
* Phân tích thành phần dinh dưỡng và nhãn thực phẩm

---

## 📂 Cấu trúc thư mục

```text
OPENFOODFACTS/
├── classification model/   # Mô hình phân loại dạng pkl
├── clustering model/       # Mô hình phân cụm dạng pkl
├── code/                   # Script xử lý dữ liệu & huấn luyện & luật kết hợp 
├── csv/                    # Dữ liệu CSV đã làm sạch
└── requirements.txt        # Thư viện Python cần thiết
```

---

## 📊 Dữ liệu

Nguồn dữ liệu: **Open Food Facts**

* Website: [https://world.openfoodfacts.org](https://world.openfoodfacts.org)
* Dữ liệu bao gồm:

Người dụng tạo thư mục `csv/` sau đó vào link drive dưới đây tải dữ liệu có tên df_final (1).csv để thực hiện các bước tiếp theo. 

Drive: [Bộ dữ liệu Open Food Facts](https://drive.google.com/drive/folders/1tcjd1UQjF6lB7EnyTZVZtTA2m6_z1Os-?fbclid=IwY2xjawO2-5BleHRuA2FlbQIxMQBzcnRjBmFwcF9pZAEwAAEeFXEiVoyrP0mDSNN_CATAbXRU0ij-oy3ChVmEx5aTmO2E8gbRIINU9bChGNY_aem_T8-mVZTmEM8aC0i89Oj23w)

---