FROM continuumio/miniconda3

# Create the environment:
COPY environment.yaml .
RUN conda env create -f environment.yaml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "geo_env", "/bin/bash", "-c"]

# Copy source code
COPY . .

# Run the application
ENTRYPOINT ["conda", "run", "-n", "geo_env", "python", "app.py"]
