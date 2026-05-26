# 1. Use an official Python runtime as the base image
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /code

# 3. Copy our requirements file first to install dependencies
COPY ./requirements.txt /code/requirements.txt

# 4. Install the Python packages listed in requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 5. Copy the rest of our application code into the container
COPY . .

# 6. Expose the port FastAPI runs on inside the container
EXPOSE 7860

# 7. Start Uvicorn, pointing to port 7860 (Hugging Face's default port)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]