FROM python:3.8

# Create the environment:
COPY requirements-app.txt .
RUN pip install -r requirements-app.txt


# Copy source code
COPY . .

# Run the application
CMD ["python", "app.py"]
