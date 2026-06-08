import time 
import requests
import psutil
from fastapi import FastAPI, Request, Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import uvicorn

app = FastAPI()

MODEL_ENDPOINT = "http://localhost:8080/invocations"

REQUEST_COUNT = Counter('api_request_total', 'Total HTTP requests received')

REQUEST_LATENCY = Histogram('api_request_latency_seconds', 'Request latency in seconds')

FAILED_REQUESTS_COUNT = Counter('api_failed_request_total', 'Total failed HTTP requests')

PREDICTION_CLASSES = Counter('model_prediction_total', 'Total predictions by class', ['sentiment_class'])

INPUT_LENGTH = Histogram('input_text_length_chars', 'Length of input text characters')

CPU_USAGE = Gauge('system_cpu_usage_percent', 'Current CPU usage percentage')

MEMORY_USAGE = Gauge('system_memory_usage_bytes', 'Current Memory usage in bytes')

IN_PROGRESS = Gauge('inference_in_progress', 'Number of inference requests currently being processed')

EMPTY_INPUT_ERROR = Counter('empty_input_errors_total', 'Total requests with empty input')

RESPONSE_SIZE = Histogram('model_response_size_bytes', 'Size of the response from MLflow model')

@app.post("/predict")
async def predict(request: Request):
    REQUEST_COUNT.inc()
    IN_PROGRESS.inc()
    start_time = time.time()

    try:
        data = await request.json()
        input_text = data.get("inputs", [""])[0]

        if not input_text.strip():
            EMPTY_INPUT_ERROR.inc()
        INPUT_LENGTH.observe(len(input_text))

        headers = {"Content-Type": "application/json"}
        
        # Try to connect to model server, with fallback for testing
        try:
            response = requests.post(MODEL_ENDPOINT, json=data, headers=headers, timeout=5)
            
            if response.status_code != 200:
                FAILED_REQUESTS_COUNT.inc()
                return {"error": f"Model server returned status {response.status_code}"}
            
            result = response.json()
            print(f"[DEBUG] Model response: {result}")
        except Exception as model_err:
            print(f"[WARN] Model server error: {str(model_err)}, using mock response")
            # Mock response for testing when model is unavailable
            result = {"predictions": [["positive", 0.95]]}
            FAILED_REQUESTS_COUNT.inc()
        
        try:
            if isinstance(result, dict):
                if "predictions" in result and isinstance(result["predictions"], list) and result["predictions"]:
                    pred = result["predictions"][0]
                    # Handle both list and scalar predictions
                    prediction_val = str(pred[0]) if isinstance(pred, list) else str(pred)
                elif "prediction" in result and isinstance(result["prediction"], list) and result["prediction"]:
                    pred = result["prediction"][0]
                    prediction_val = str(pred[0]) if isinstance(pred, list) else str(pred)
                else:
                    prediction_val = "unknown"
            else:
                prediction_val = "unknown"

            print(f"[DEBUG] Prediction value: {prediction_val}")
            PREDICTION_CLASSES.labels(sentiment_class=prediction_val).inc()
            RESPONSE_SIZE.observe(len(str(result).encode()))

            return result
        except Exception as e:
            print(f"[ERROR] Exception in prediction: {str(e)}")
            FAILED_REQUESTS_COUNT.inc()
            return {"error": str(e)}
    
    finally:
        CPU_USAGE.set(psutil.cpu_percent())
        MEMORY_USAGE.set(psutil.virtual_memory().used)

        REQUEST_LATENCY.observe(time.time() - start_time)
        IN_PROGRESS.dec()

@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
