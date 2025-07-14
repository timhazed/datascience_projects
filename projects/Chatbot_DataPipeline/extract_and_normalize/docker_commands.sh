# Normal build
docker build -t extract-normalize .

# No cache build
docker build --no-cache -t extract-normalize .

# Example commands to run the container
docker run \
  --env-file ~/src/python/.env_mini \
  -v /Users/timhazed/tmp/input:/app/input \
  -v /Users/timhazed/tmp/output_test:/app/output \
  -e PYTHONUNBUFFERED=1 \
  extract-normalize \
  --url "https://www.mdpi.com/2673-4591/9/1/10/pdf" \
  --output "/app/output" \
  --config "config/config.yaml"