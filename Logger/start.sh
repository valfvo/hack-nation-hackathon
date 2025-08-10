docker run -it logger \
  -v "$(pwd)":/app \  # mount your current local folder into /app
-w /app \             # set container's working directory
bash
