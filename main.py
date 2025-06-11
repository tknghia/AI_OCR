import sys
import os

# Thêm thư mục src vào PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), 'View'))

# Bây giờ có thể import module từ src
from View import view

if __name__ == "__main__":
    view.start_app()