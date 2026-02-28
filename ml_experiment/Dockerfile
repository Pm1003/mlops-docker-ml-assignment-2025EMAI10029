# Use Python base image
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run ML script
CMD ["python", "palmer-panguin-decisionTree.py"]