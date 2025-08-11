# Use nginx to serve a simple static page for demo
FROM nginx:alpine

# Create a simple HTML page
RUN echo '<html><head><title>Yubu Code - Frontend</title></head><body><h1>Yubu Code Frontend</h1><p>This is a placeholder frontend for the Docker Compose demo.</p><p>Backend API: <a href="http://localhost:8000/health">http://localhost:8000/health</a></p><p>Logger API: <a href="http://localhost:8001">http://localhost:8001</a></p></body></html>' > /usr/share/nginx/html/index.html

# Expose port 80 (nginx default) and map to 3000 in docker-compose
EXPOSE 80