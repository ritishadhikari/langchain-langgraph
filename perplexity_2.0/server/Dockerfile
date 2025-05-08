FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose Port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn","app:app", "--host", "0.0.0.0", "--port", "8000"]

