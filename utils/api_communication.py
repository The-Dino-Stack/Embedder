import logging
import requests

from urllib.parse import urljoin

def check_api_health(api_url, api_key):
    try:
        headers = {"x-api-key": api_key}
        response = requests.get(urljoin(api_url, "/health"), headers=headers)
        if response.status_code == 200:
            logging.info("✅ API is reachable.")
            return True
        else:
            logging.error(f"❌ API health check failed: {response.status_code} {response.text}")
            return False
    except requests.RequestException as e:
        logging.error(f"❌ API health check failed: {e}")
        return False

def upload_embeddings(api_url, api_key, embeddings):
    try:
        headers = {"x-api-key": api_key, "Content-Type": "application/json"}
        response = requests.post(
            urljoin(api_url, "/upload-embeddings"),
            headers=headers,
            json=embeddings
        )
        if response.status_code == 200:
            logging.info("✅ Embeddings uploaded successfully.")
        else:
            logging.error(f"❌ Failed to upload embeddings: {response.status_code} {response.text}")
    except requests.RequestException as e:
        logging.error(f"❌ Failed to upload embeddings: {e}")