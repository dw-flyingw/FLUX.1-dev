FROM python:3.11-slim

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
    fastapi>=0.115.0 \
    uvicorn[standard]>=0.34.0 \
    jinja2>=3.1.0 \
    requests>=2.31.0 \
    pillow>=10.2.0 \
    python-dotenv>=1.0.0 \
    pydantic>=2.6.0 \
    pydantic-settings>=2.2.0 \
    openai>=1.12.0 \
    python-multipart>=0.0.9

COPY . .

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
