# Use official Python image (slim version to keep it light)
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements.txt first (if you have one)
# If you don't have a requirements.txt, you can create one or list dependencies directly in the Dockerfile.
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire app code
COPY . .

# Expose FastAPI default port
EXPOSE 8000

# Command to run the app with uvicorn
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]