#!/bin/bash
# Shared devcontainer post-start setup
# Called every time the container starts (including restarts).
# Opens the Docker socket so the dev user can access Docker
# regardless of the host's docker group ID.
# This is safe because devcontainers already run --privileged.

if [ -S /var/run/docker.sock ]; then
    sudo chmod 666 /var/run/docker.sock
fi
