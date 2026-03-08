FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860

WORKDIR /app

COPY server/requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && pip install -r /tmp/requirements.txt

COPY . /app
RUN pip install -e .

EXPOSE 7860

CMD ["python", "-m", "dfa_agent_env.server.app", "--host", "0.0.0.0", "--port", "7860"]
