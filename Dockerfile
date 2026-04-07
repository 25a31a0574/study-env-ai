FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Expose port for FastAPI (standard for Hugging Face Spaces)
EXPOSE 7860

# Run the application
CMD ["uvicorn", "inference.py:app", "--host", "0.0.0.0", "--port", "7860"]
