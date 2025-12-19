#Use a stable Python version that matches sklearn compatibility
FROM python:3.12-slim

#Set working directory
WORKDIR /app

#Copy requirements first (better Docker caching)
COPY requirements.txt .

#Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

#Copy application code and models
COPY . .

#Expose Flask port
EXPOSE 5000

#Run the Flask app
CMD ["gunicorn", "-b", "0.0.0.0:5000", "wsgi:create_app()"]