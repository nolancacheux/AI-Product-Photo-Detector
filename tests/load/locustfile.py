"""Locust load test for AI Product Photo Detector API."""

import io

from locust import HttpUser, between, task
from PIL import Image


class APIUser(HttpUser):
    """Simulates a typical API consumer."""

    wait_time = between(1, 3)

    def on_start(self):
        self.headers = {"X-API-Key": "test-key"}
        img = Image.new("RGB", (224, 224), "blue")
        buf = io.BytesIO()
        img.save(buf, "JPEG")
        self.image_bytes = buf.getvalue()

    @task(1)
    def health(self):
        self.client.get("/health")

    @task(5)
    def predict(self):
        self.client.post(
            "/predict",
            headers=self.headers,
            files={"file": ("test.jpg", self.image_bytes, "image/jpeg")},
        )

    @task(1)
    def metrics(self):
        self.client.get("/metrics")
