FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY=localhost,127.0.0.1
ARG SSL_CERT_FILE=/etc/ssl/certs/ca-bundle-combined.pem
ARG REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-bundle-combined.pem
ARG CURL_CA_BUNDLE=/etc/ssl/certs/ca-bundle-combined.pem
ENV HTTP_PROXY=${HTTP_PROXY} \
    HTTPS_PROXY=${HTTPS_PROXY} \
    NO_PROXY=${NO_PROXY} \
    http_proxy=${HTTP_PROXY} \
    https_proxy=${HTTPS_PROXY} \
    no_proxy=${NO_PROXY} \
    SSL_CERT_FILE=${SSL_CERT_FILE} \
    REQUESTS_CA_BUNDLE=${REQUESTS_CA_BUNDLE} \
    CURL_CA_BUNDLE=${CURL_CA_BUNDLE}

RUN pip install --no-cache-dir \
    --trusted-host pypi.org \
    --trusted-host pypi.python.org \
    --trusted-host files.pythonhosted.org \
    "diffusers>=0.31.0" \
    "transformers>=4.40.0" \
    "accelerate>=0.30.0" \
    "safetensors>=0.4.0" \
    "fastapi>=0.115.0" \
    "uvicorn[standard]>=0.30.0" \
    "sentencepiece>=0.2.0" \
    "protobuf>=5.27.0" \
    "pillow>=10.2.0"

COPY inference/server.py server.py
COPY inference/ip_adapter_attention.py ip_adapter_attention.py

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
