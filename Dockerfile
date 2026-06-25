FROM python:3.10-slim

WORKDIR /app


COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

# Hugging Face Spaces listen on port 7860 by default
EXPOSE 7860

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
