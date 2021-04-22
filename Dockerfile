FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

COPY ./cassava /cassava
COPY app.py app.py 
ADD ./requirements.txt requirements.txt
RUN pip install -r requirements.txt


ENV PYTHONUNBUFFERED=TRUE
EXPOSE 8003
ENTRYPOINT gunicorn app:app --bind 0.0.0.0:8003 --timeout 1200 -w 1 -k uvicorn.workers.UvicornWorker