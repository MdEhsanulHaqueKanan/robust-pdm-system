# --- Stage 1: Build the Python Environment and Install Dependencies ---
FROM python:3.10-slim-bullseye AS build_env

WORKDIR /build

COPY requirements.txt .

# Install ONLY Python dependencies here. System libs will be installed in the final stage.
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- Stage 2: Create the Final Application Image (Runtime) ---
FROM python:3.10-slim-bullseye

# Set the working directory for the application
WORKDIR /app

# --- CRITICAL FIX: Install system dependencies here, in the final stage ---
# libgomp1 is for OpenMP parallelism (needed by LightGBM/XGBoost).
# libstdc++6 is a standard C++ library often implicitly required.
# gcc is sometimes needed for runtime components for dynamically linked libraries.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgomp1 \
    libstdc++6 \
    gcc && \
    rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from the build_env stage.
COPY --from=build_env /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# Copy the rest of your application code (source files, models, data)
COPY . .

# Set environment variables for Flask
ENV FLASK_APP=run.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_DEBUG=False

# Ensure models and processed data directories exist.
RUN mkdir -p ml_models/rul_model \
             ml_models/classification_model \
             data/processed \
             data/raw/C-MAPSS

# Expose the port your Flask app will run on.
EXPOSE 5000

# The command to run your Flask application when the container starts.
CMD ["python", "-m", "flask", "run"]