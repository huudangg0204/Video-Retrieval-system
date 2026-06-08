# Sử dụng image Python nhẹ
FROM python:3.10-slim

# Cài đặt gói hệ thống cần thiết
RUN apt-get update && apt-get install -y \
    git ffmpeg libsm6 libxext6 wget unzip && \
    rm -rf /var/lib/apt/lists/*

# Tạo thư mục app
WORKDIR /app

# Copy file requirements trước để cache install
COPY requirements.txt .

# Cài dependencies
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Copy toàn bộ project
COPY . .

# Expose port cho Heroku
EXPOSE 8080

# Chạy bằng gunicorn (Heroku sẽ map $PORT tự động)
CMD gunicorn --bind 0.0.0.0:$PORT server:app
