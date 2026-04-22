"""Main entry point to run the Flask server."""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.api.app import create_app

if __name__ == '__main__':
    app = create_app()
    print("Starting 云南省企业就业失业数据采集系统 (Agent-based)")
    print("API server: http://localhost:5000")
    print("Health check: GET http://localhost:5000/api/health")
    app.run(host='0.0.0.0', port=5000, debug=False)
