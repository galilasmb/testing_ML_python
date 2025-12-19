#!/bin/bash
# Simple script to start the FastAPI server

echo "Starting Image Inference Service..."
echo "=================================="
echo ""
echo "Server will be available at: http://localhost:8000"
echo "API documentation at: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
