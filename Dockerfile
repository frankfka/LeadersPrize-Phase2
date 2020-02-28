FROM tensorflow/tensorflow:latest-gpu-py3

# Basic setup
RUN pip install --upgrade pip

# Make the working directory
RUN mkdir -p /usr/local
WORKDIR /usr/local

# Install dependencies
COPY requirements.txt .
RUN pip install --trusted-host pypi.python.org -r requirements.txt --timeout 1000

# Install NLTK requirements
RUN [ "python3", "-c", "import nltk; nltk.download('popular')" ]

# Copy all files over
COPY . .

# Run command to start predicting
CMD [ "python3", "main.py" ]