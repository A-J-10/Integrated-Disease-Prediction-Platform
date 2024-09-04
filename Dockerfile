FROM python:latest

WORKDIR /root

COPY . .

RUN pip install -r requirements.txt

EXPOSE 8501

CMD [ "streamlit","run","app.py" ]

