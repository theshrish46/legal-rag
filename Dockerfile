# 1. Use a slim image to keep it under 200MB
FROM python:3.12-slim

# 2. Prevent Python from buffering stdout (Essential for seeing logs!)
ENV PYTHONUNBUFFERED=1

# 3. Set work directory
WORKDIR /app

# 4. Install system dependencies (Legal-RAG sometimes needs these for PDF parsing)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy your project
COPY . .

# 7. Run your script
CMD ["streamlit", "run", "app.py"]