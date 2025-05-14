FROM python:3.10-slim
    
WORKDIR /app

COPY requirements.txt .
    
RUN pip install --upgrade pip
   
RUN pip install --no-cache-dir -r requirements.txt --no-deps

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

RUN apt update && apt install pandoc -y

RUN apt-get install texlive-latex-base texlive-latex-extra texlive-fonts-recommended -y
   
ENV TOKENIZERS_PARALLELISM=false

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]