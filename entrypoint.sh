#!/bin/bash

export PATH="$PATH:/home/devbox/.local/bin"

app_env=${1:-development}

# Development environment commands
dev_commands() {
    echo "Running development environment commands..."
    streamlit run app.py --server.port 8080 --server.address 0.0.0.0
}

# Production environment commands
prod_commands() {
    echo "Running production environment commands..."
    streamlit run app.py --server.port 8080 --server.address 0.0.0.0 --server.headless true
}

# Check environment variables to determine the running environment
if [ "$app_env" = "production" ] || [ "$app_env" = "prod" ] ; then
    echo "Production environment detected"
    prod_commands
else
    echo "Development environment detected"
    dev_commands
fi
