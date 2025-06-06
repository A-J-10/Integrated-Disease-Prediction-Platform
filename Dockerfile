FROM python:3.11-slim

WORKDIR /root

COPY . .

RUN pip install -r requirements.txt

EXPOSE 8501

CMD [ "streamlit","run","app.py" ]

