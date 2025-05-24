FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y \
  gcc \
  libpq-dev \
  && rm -rf /var/lib/apt/lists/*

# Install PostgreSQL client (version 17 to match server)
RUN apt-get update && apt-get install -y gnupg2 lsb-release && \
  echo "deb http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list && \
  apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 7FCC7D46ACCC4CF8 && \
  apt-get update && apt-get install -y postgresql-client-17 && \
  rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/app .
COPY common_utils/ ./common_utils/

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]