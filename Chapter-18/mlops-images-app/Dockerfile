
# Use an official Python runtime as a parent image
FROM asia-docker.pkg.dev/vertex-ai/training/tf-cpu.2-12.py310:latest

WORKDIR /

COPY requirements.txt /requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copies the trainer code to the Docker image.
COPY trainer /trainer

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "-m", "trainer.train"]
