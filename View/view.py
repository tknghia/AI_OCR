
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__,template_folder='.')

@app.route('/')
def index():
    return render_template('./index.html',)

#@app.route('/convert', methods=['POST'])
# def convert():
#     # Xử lý file upload và chuyển đổi tại đây
#     file = request.files['file']
#     # Thêm logic chuyển đổi ở đây (sử dụng thư viện phù hợp)
#     return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
