FROM tiangolo/uvicorn-gunicorn-fastapi:python3.11
ADD ./model.pkl /app/model.pkl
ADD ./mlapi.py /app/mlapi.py
COPY ./requirements.txt ./
RUN pip install -r ./requirements.txt
COPY . . 
EXPOSE 8000