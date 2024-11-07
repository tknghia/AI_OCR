from docx import Document

# Hàm để đánh số lại các "Samples" và ghi đè lên file cũ
def renumber_samples(doc_path):
    # Mở file Word
    doc = Document(doc_path)
    
    # Biến đếm cho các sample
    sample_count = 1
    
    # Duyệt qua các đoạn (paragraphs) trong tài liệu
    for paragraph in doc.paragraphs:
        if paragraph.text.startswith("Samples - "):
            # Tìm các đoạn văn bắt đầu với "Samples - " và thay thế bằng số đúng
            new_text = f"Samples - {sample_count}"
            paragraph.text = new_text
            sample_count += 1
    
    # Ghi đè file Word cũ
    doc.save(doc_path)
    print(f"Đã lưu tài liệu với các số Samples được đánh lại tại: {doc_path}")

# Đường dẫn tới file Word
input_file = "Logs.docx"  # Đổi thành đường dẫn của bạn

# Gọi hàm để đánh số lại và ghi đè file cũ
renumber_samples(input_file)
