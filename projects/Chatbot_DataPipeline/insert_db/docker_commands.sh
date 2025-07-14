# No cache build
docker build --no-cache -t insert-db .

# Build it with sentence transformer preloaded
docker build --no-cache -f Dockerfile_WithTransformer -t insert-db .

# Example commands to run the container
docker run \
  --env-file ~/src/python/.env_mini \
  -v /Users/timhazed/tmp/demo/output_demo/chunked:/app/input \
  -e PYTHONUNBUFFERED=1 \
  insert-db \
  --input "/app/input/doc_0002"