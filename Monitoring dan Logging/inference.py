import requests
import time
import random

PROXY_URL = "http://localhost:8002/predict" 

# Simulasi data ulasa
sample_reviews = [
    "I absolutly love this product, it is amazing!",
    "Terrible quality, broke after one day of use.",
    "It is okay, nothing special but gets the job done.",
    "Highly recommended, fast shipping and great price.",
    "",
    "Worst customer service ever, do not buy this."
]

def simulate_traffic():
    while True:
        review = random.choice(sample_reviews)
        payload = {"inputs": [review]}

        try:
            response = requests.post(PROXY_URL, json=payload)
            print(f"Sent: {review[:20]}... | Status: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Koneksi gagal: {e}")

        time.sleep(random.uniform(0.5, 2.0))

if __name__ == "__main__":
    simulate_traffic()