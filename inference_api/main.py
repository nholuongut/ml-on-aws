
import fastapi
import uvicorn
import os

import logging
import boto3

from src import app
api = fastapi.FastAPI()

def configure():
    configure_routing()

def configure_routing():
    api.include_router(app.router)


if __name__ == '__main__':
    configure()
    uvicorn.run(api, port=5000, host='127.0.0.1')
else:
    configure()