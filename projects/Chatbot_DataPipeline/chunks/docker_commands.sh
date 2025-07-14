# No cache build
docker build --no-cache -t chunker .

# Example commands to run the container
docker run \
  --env-file ~/src/python/.env_mini \
  -v /Users/timhazed/tmp/output:/app/input \
  -e PYTHONUNBUFFERED=1 \
  chunker \
  --input "/app/input/extracted/doc_0001"