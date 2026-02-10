/**
 * k6 load test for AI Product Photo Detector API.
 *
 * Usage:
 *   k6 run tests/load/k6_test.js
 *   k6 run --vus 50 --duration 2m tests/load/k6_test.js
 */

import http from "k6/http";
import { check, sleep } from "k6";
import { Rate, Trend } from "k6/metrics";
import { randomIntBetween } from "https://jslib.k6.io/k6-utils/1.4.0/index.js";

const BASE_URL = __ENV.BASE_URL || "http://localhost:8000";
const API_KEY = __ENV.API_KEY || "test-key";

// Custom metrics
const errorRate = new Rate("errors");
const predictionDuration = new Trend("prediction_duration", true);

// Test configuration
export const options = {
  scenarios: {
    smoke: {
      executor: "constant-vus",
      vus: 5,
      duration: "30s",
      tags: { scenario: "smoke" },
    },
    load: {
      executor: "ramping-vus",
      startVUs: 0,
      stages: [
        { duration: "30s", target: 20 },
        { duration: "1m", target: 20 },
        { duration: "30s", target: 50 },
        { duration: "1m", target: 50 },
        { duration: "30s", target: 0 },
      ],
      startTime: "35s",
      tags: { scenario: "load" },
    },
    spike: {
      executor: "ramping-vus",
      startVUs: 0,
      stages: [
        { duration: "10s", target: 100 },
        { duration: "30s", target: 100 },
        { duration: "10s", target: 0 },
      ],
      startTime: "4m30s",
      tags: { scenario: "spike" },
    },
  },
  thresholds: {
    http_req_duration: ["p(95)<2000", "p(99)<5000"],
    errors: ["rate<0.1"],
    prediction_duration: ["p(95)<3000"],
  },
};

// Generate a minimal JPEG image in k6 (1x1 blue pixel)
// Pre-encoded minimal JPEG for performance
const JPEG_HEADER = open("../../tests/data/sample_real.jpg", "b");

export default function () {
  const scenario = randomIntBetween(1, 7);

  if (scenario <= 1) {
    healthCheck();
  } else if (scenario <= 6) {
    predict();
  } else {
    metricsEndpoint();
  }

  sleep(randomIntBetween(1, 3));
}

function healthCheck() {
  const res = http.get(`${BASE_URL}/health`);
  const passed = check(res, {
    "health status 200": (r) => r.status === 200,
    "health has status field": (r) => {
      try {
        return JSON.parse(r.body).status !== undefined;
      } catch {
        return false;
      }
    },
  });
  errorRate.add(!passed);
}

function predict() {
  const payload = {
    file: http.file(JPEG_HEADER, "test.jpg", "image/jpeg"),
  };

  const params = {
    headers: {
      "X-API-Key": API_KEY,
    },
    tags: { endpoint: "predict" },
  };

  const res = http.post(`${BASE_URL}/predict`, payload, params);
  predictionDuration.add(res.timings.duration);

  const passed = check(res, {
    "predict status 200": (r) => r.status === 200,
    "predict has prediction": (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.prediction !== undefined || body.result !== undefined;
      } catch {
        return false;
      }
    },
  });
  errorRate.add(!passed);
}

function metricsEndpoint() {
  const res = http.get(`${BASE_URL}/metrics`);
  const passed = check(res, {
    "metrics status 200": (r) => r.status === 200,
    "metrics has prometheus format": (r) =>
      r.body && r.body.includes("aidetect_"),
  });
  errorRate.add(!passed);
}
