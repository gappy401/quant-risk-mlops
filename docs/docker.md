# Docker — Packaging the Service

> Rundown page for the **Docker** layer of `quant-risk-mlops`, in the same shape as the other technology pages.

## What Docker is

Docker packages an application together with everything it needs to run (the code, the Python runtime, the libraries, the settings) into a single portable unit called an **image**. A **container** is a running instance of that image. Because the image carries its own environment, it runs the same on your laptop, in CI, and on a server. This is the answer to "it works on my machine."

## What it does

1. **Builds images** from a recipe file (the Dockerfile).
2. **Runs containers** from those images, isolated from the host and from each other.
3. **Maps ports** so a service inside a container is reachable from outside.
4. **Shares images** through registries (Docker Hub, GHCR, ECR).
5. **Composes multi container stacks** so several services run and talk to each other with one command.

## Core concepts

1. **Image**: the built, read only package. Built once, run many times.
2. **Container**: a running image.
3. **Dockerfile**: the build recipe.
4. **Layers**: each Dockerfile instruction adds a cached layer. Unchanged layers are reused, which is why instruction order affects build speed.
5. **Registry**: where images are stored and pulled from.
6. **Volume**: storage that outlives a container.
7. **Compose**: a YAML file describing several services as one stack.

## Basic Dockerfile syntax

Each line is an instruction that becomes a layer:

```dockerfile
FROM python:3.12-slim          # the base image to build on
WORKDIR /app                   # the working directory inside the image
COPY requirements-serve.txt .  # copy files from your machine into the image
RUN pip install -r requirements-serve.txt   # run a command at build time
ENV PYTHONUNBUFFERED=1         # set an environment variable
EXPOSE 8000                    # document the port the app listens on
USER appuser                   # run as a non root user
HEALTHCHECK CMD ...            # how Docker checks the container is healthy
CMD ["uvicorn", "quant_risk.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]  # the default start command
```

## The commands (and how to read them)

The pattern is `docker VERB [OPTIONS] [ARGS]`.

1. `docker build -t credit-pd-api -f docker/Dockerfile.api .` builds an image. The `-t` flag names (tags) it, `-f` points at the Dockerfile, and the final `.` is the build context (the folder whose files can be copied in).
2. `docker run --rm -p 8000:8000 credit-pd-api` runs a container. `-p host:container` maps ports, `--rm` deletes the container when it stops.
3. `docker ps` lists running containers, `docker images` lists images.
4. `docker logs CONTAINER` prints a container's output.
5. `docker exec -it CONTAINER sh` opens a shell inside a running container.
6. `docker compose -f docker/docker-compose.yml up -d` starts a multi service stack in the background, `down` stops it, `restart SERVICE` restarts one service.

## How it fits this project, and where it comes alive

The serving API is packaged by `docker/Dockerfile.api` into a lean image: it starts from `python:3.12-slim`, installs only the serving dependencies from `requirements-serve.txt` (no PySpark or DVC, so the image stays small), installs the project package, runs as a non root user, and declares a `HEALTHCHECK` that calls `/health`. The dependency layer is copied and installed before the source code, so editing a `.py` file does not trigger a slow reinstall. The start command binds uvicorn to `0.0.0.0` so the port is reachable from outside the container.

Then `docker/docker-compose.yml` runs two services together: the API and an MLflow tracking server. The API reaches the server by its service name (`http://mlflow:5000`), which is how the model gets loaded from a registry running in a separate container.

Where it comes alive is the first containerized response. The moment `docker run` served `/health` with a 200 from inside the container, the packaging worked. Then under compose, once the MLflow server was reachable and the model loaded, `/ready` turned green and the API scored a real request entirely inside containers. That same image is what later runs on Kubernetes unchanged, which is the whole portability promise.

## Documentation links

1. Get started: https://docs.docker.com/get-started/
2. Dockerfile reference: https://docs.docker.com/reference/dockerfile/
3. Image building best practices: https://docs.docker.com/build/building/best-practices/
4. Docker Compose: https://docs.docker.com/compose/
