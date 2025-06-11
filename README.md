Lỗi bạn gặp là do đoạn `README.md` của bạn đang dùng **Markdown sai cú pháp**, cụ thể là bạn mở các khối code (\`\`\`bash) nhưng **không đóng chúng lại**, và **viết Markdown link trong khối code**.

---

## ✅ Dưới đây là bản sửa **đúng cú pháp** cho đoạn README của bạn:

````markdown
# AI_OCR 

### 1. Cài đặt thư viện Node.js
```bash
npm install
````

### 2. Cài đặt các thư viện Python trong requirements.txt

```bash
pip install -r requirements.txt
```

### 3. Tải file `.pth` từ Google Drive

[Link tải file `.pth`](https://drive.google.com/file/d/1bqW5GywnoWtlqlft_t4MVzk_Pxh1eZxT/view?usp=sharing)

Sau khi tải, hãy đặt nó vào thư mục:

```bash
Model/data/
```

### 4. Chạy ứng dụng

```bash
python main.py
```

````

---

## 📌 Lưu ý quan trọng:
- **Mỗi khối lệnh (`bash`) phải được đóng bằng ` ``` `** — nếu không, Markdown sẽ bị hỏng.
- **Link (`[text](url)`) không được viết bên trong khối lệnh (`bash`)**.

---

Bạn có muốn mình giúp tạo `README.md` hoàn chỉnh, có cả cấu trúc thư mục, mô tả API hoặc hình ảnh minh họa?
````

