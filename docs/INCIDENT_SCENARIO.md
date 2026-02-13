# Incident Response Playbook

## Table of Contents

1. [Severity Levels](#severity-levels)
2. [Escalation Matrix](#escalation-matrix)
3. [Incident Response Process](#incident-response-process)
4. [Runbooks](#runbooks)
5. [Incident Scenario: Data Drift](#incident-scenario-data-drift)
6. [Post-Mortem Template](#post-mortem-template)

---

## Severity Levels

| Level | Name | Description | Response Time | Examples |
|-------|------|-------------|---------------|----------|
| **P1** | Critical | Service is down or completely unusable. Data loss or security breach. | **15 min** | API returning 5xx for all requests, model not loaded, security breach |
| **P2** | High | Major feature degraded. Significant impact on users. | **1 hour** | Error rate > 5%, P95 latency > 5s, drift score > 0.30 |
| **P3** | Medium | Minor feature degraded. Limited user impact. | **4 hours** | Drift score > 0.15, memory growing steadily, rate limiting triggered |
| **P4** | Low | Cosmetic issue or minor inconvenience. No user impact. | **24 hours** | Dashboard missing data, non-critical log warnings |

### Severity Decision Tree

```
Is the service completely down?
├── YES → P1
└── NO → Are >5% of requests failing?
    ├── YES → P2
    └── NO → Is core functionality degraded?
        ├── YES → P3
        └── NO → P4
```

---

## Escalation Matrix

| Severity | First Responder | Escalation (30 min) | Escalation (2 hours) |
|----------|-----------------|----------------------|----------------------|
| **P1** | On-call engineer | Team lead + second engineer | Engineering manager |
| **P2** | On-call engineer | Team lead | Engineering manager (if unresolved 4h) |
| **P3** | On-call engineer | — | Team lead (next business day) |
| **P4** | Assigned engineer | — | — |

### Team Responsibilities

| Team | Scope | Alerts Owned |
|------|-------|-------------|
| **Backend** | API server, HTTP layer, rate limiting | HighErrorRate, HighLatency, HighRateLimiting |
| **ML Engineering** | Model inference, predictions, drift | DriftDetected, DriftCritical, ModelNotLoaded, HighPredictionErrorRate |
| **Infrastructure** | Runtime, containers, resources | ServiceDown, HighMemoryUsage, HighCPUUsage, HighFileDescriptorUsage |

---

## Incident Response Process

### 1. Detection

- Prometheus alert fires → notification sent to on-call
- User report received through support channel
- Anomaly spotted on Grafana dashboard

### 2. Acknowledge

- Acknowledge the alert within the response time SLA
- Create an incident channel/thread for communication
- Post initial status: what is known so far

### 3. Triage

- Determine severity level using the decision tree
- Identify which runbook applies
- Check recent deployments or changes (last 24h)

### 4. Mitigate

- Apply the relevant runbook steps
- Focus on restoring service first, root cause later
- Communicate status updates every 30 minutes (P1) or every hour (P2)

### 5. Resolve

- Confirm the issue is resolved via metrics
- Monitor for 30 minutes to ensure stability
- Close the incident

### 6. Post-Mortem

- Schedule within 48 hours of resolution (P1/P2)
- Use the template below
- Track action items to completion

---

## Runbooks

### Runbook: High Error Rate

**Alert:** `HighErrorRate` — 5xx error rate exceeds 5% for 5 minutes.

**Dashboard:** [API Performance](http://localhost:3000/d/ai-detector-api)

**Diagnosis Steps:**

1. Check which endpoints are failing:
   ```bash
   # Check error breakdown by endpoint
   curl -s http://localhost:9090/api/v1/query \
     --data-urlencode 'query=sum(rate(aidetect_http_requests_total{status_code=~"5.."}[5m])) by (endpoint)' \
     | python -m json.tool
   ```

2. Check API logs for error details:
   ```bash
   docker compose logs api --tail=100 | grep -i "error\|exception\|traceback"
   ```

3. Check if the model is loaded:
   ```bash
   curl http://localhost:8080/health
   ```

4. Check resource utilization:
   ```bash
   docker stats --no-stream
   ```

**Resolution Steps:**

| Cause | Action |
|-------|--------|
| Model not loaded | Restart the API container: `docker compose restart api` |
| Out of memory | Increase memory limit or reduce batch sizes |
| Upstream dependency failure | Check network connectivity, restart affected services |
| Code bug (after deployment) | Rollback: `docker compose up -d --force-recreate api` with previous image |

---

### Runbook: High Latency

**Alert:** `HighLatency` — P95 latency exceeds 2 seconds for 5 minutes.

**Dashboard:** [API Performance](http://localhost:3000/d/ai-detector-api)

**Diagnosis Steps:**

1. Check latency by endpoint:
   ```bash
   curl -s http://localhost:9090/api/v1/query \
     --data-urlencode 'query=histogram_quantile(0.95, sum(rate(aidetect_http_request_duration_seconds_bucket[5m])) by (le, endpoint))' \
     | python -m json.tool
   ```

2. Check concurrent request count:
   ```bash
   curl -s http://localhost:9090/api/v1/query \
     --data-urlencode 'query=aidetect_active_requests' \
     | python -m json.tool
   ```

3. Check CPU and memory pressure:
   ```bash
   docker stats --no-stream
   ```

4. Check image sizes (large images = slow inference):
   ```bash
   curl -s http://localhost:9090/api/v1/query \
     --data-urlencode 'query=histogram_quantile(0.95, sum(rate(aidetect_image_size_bytes_bucket[5m])) by (le))' \
     | python -m json.tool
   ```

**Resolution Steps:**

| Cause | Action |
|-------|--------|
| High traffic spike | Scale horizontally or enable rate limiting |
| Large images | Enforce stricter image size limits |
| CPU saturation | Scale vertically (more CPU) or horizontally (more replicas) |
| Memory pressure (GC) | Check GC metrics, consider memory optimization |
| Slow model inference | Profile inference pipeline, consider model optimization |

---

### Runbook: Service Down

**Alert:** `ServiceDown` — Health check fails for 1 minute.

**Dashboard:** [Infrastructure](http://localhost:3000/d/ai-detector-infra)

**Diagnosis Steps:**

1. Check container status:
   ```bash
   docker compose ps
   ```

2. Check container logs:
   ```bash
   docker compose logs api --tail=200
   ```

3. Check if the process is running:
   ```bash
   docker compose exec api ps aux
   ```

4. Check system resources:
   ```bash
   df -h          # Disk space
   free -h        # System memory
   docker stats   # Container resources
   ```

**Resolution Steps:**

| Cause | Action |
|-------|--------|
| Container crashed | `docker compose up -d api` |
| OOM killed | Increase memory limits, check for leaks |
| Disk full | Clean up logs/temp files, increase disk |
| Port conflict | Check `docker compose ps`, resolve conflicts |
| Configuration error | Check env vars, config files, rollback if recent change |

**Immediate Mitigation:**
```bash
# Quick restart
docker compose restart api

# Full recreate
docker compose up -d --force-recreate api

# Check health after restart
sleep 10 && curl http://localhost:8080/health
```

---

### Runbook: Drift Detected

**Alert:** `DriftDetected` — Drift score exceeds 0.15 for 10 minutes.

**Dashboard:** [Model Performance](http://localhost:3000/d/ai-detector-model)

**Diagnosis Steps:**

1. Check current drift status:
   ```bash
   curl http://localhost:8080/monitoring/drift
   ```

2. Analyze prediction distribution:
   ```bash
   # Check AI vs Real ratio
   curl -s http://localhost:9090/api/v1/query \
     --data-urlencode 'query=sum(rate(aidetect_predictions_total{status="success"}[1h])) by (prediction)' \
     | python -m json.tool
   ```

3. Check confidence levels:
   ```bash
   curl -s http://localhost:9090/api/v1/query \
     --data-urlencode 'query=sum(rate(aidetect_predictions_total{status="success"}[1h])) by (confidence)' \
     | python -m json.tool
   ```

4. Sample low-confidence predictions for manual review.

**Resolution Steps:**

| Drift Score | Action |
|-------------|--------|
| 0.15 – 0.25 | Monitor closely, begin data collection for retraining |
| 0.25 – 0.40 | Lower decision threshold (0.5 → 0.35) as temporary measure, fast-track retraining |
| > 0.40 | **Critical:** Enable human-in-the-loop for low-confidence predictions, emergency retraining |

**Retraining Process:**
```bash
# 1. Collect new training data
# 2. Retrain model
python -m src.training.train --config configs/train_config.yaml

# 3. Validate on held-out set
# 4. Deploy with canary (10% traffic)
# 5. Monitor drift score — should decrease
# 6. Full rollout
# 7. Update baseline
```

---

### Runbook: High Memory Usage

**Alert:** `HighMemoryUsage` — RSS exceeds 80% of limit for 5 minutes.

**Dashboard:** [Infrastructure](http://localhost:3000/d/ai-detector-infra)

**Diagnosis Steps:**

1. Check current memory usage:
   ```bash
   docker stats --no-stream --format "table {{.Name}}\t{{.MemUsage}}\t{{.MemPerc}}"
   ```

2. Check memory growth rate:
   ```bash
   curl -s http://localhost:9090/api/v1/query \
     --data-urlencode 'query=deriv(process_resident_memory_bytes{job="ai-detector-api"}[30m])' \
     | python -m json.tool
   ```

3. Check GC activity:
   ```bash
   curl -s http://localhost:9090/api/v1/query \
     --data-urlencode 'query=rate(python_gc_collections_total{job="ai-detector-api"}[5m])' \
     | python -m json.tool
   ```

**Resolution Steps:**

| Cause | Action |
|-------|--------|
| Memory leak | Restart service as mitigation, investigate code |
| Large batch requests | Enforce smaller batch size limits |
| Model loaded multiple times | Check singleton pattern, restart service |
| Image processing buffers | Verify buffers are freed after use |

**Immediate Mitigation:**
```bash
# Graceful restart to reclaim memory
docker compose restart api
```

---

## Incident Scenario: Data Drift

### Context

The AI Product Photo Detector has been running in production for 6 months, serving an e-commerce platform with ~50,000 image classifications per day. The model was trained on a dataset containing AI-generated images from **Stable Diffusion 1.5/2.1, DALL-E 2, and Midjourney v4/v5**.

### Triggering Event

In Q1 2026, a new wave of AI image generators is released:
- **DALL-E 3** (improved photorealism, better text rendering)
- **Midjourney v6** (near-photographic quality, improved lighting)
- **Stable Diffusion XL Turbo** (faster generation with higher fidelity)

These newer models produce images with fundamentally different characteristics:
- Higher resolution and sharper details
- More accurate lighting and shadows
- Fewer typical AI artifacts (distorted hands, blurred text)
- Better color consistency and natural backgrounds

### Impact

| Metric | Before Drift | After Drift | Change |
|--------|-------------|-------------|--------|
| **Accuracy** | 83.2% | 65.1% | -18.1% |
| **Precision (AI class)** | 85.7% | 58.3% | -27.4% |
| **Recall (AI class)** | 80.1% | 71.2% | -8.9% |
| **False Negative Rate** | 19.9% | 41.7% | +21.8% |
| **Low Confidence Ratio** | 12.3% | 34.8% | +22.5% |

The model increasingly **misclassifies newer AI-generated images as real**, allowing fraudulent product listings to pass undetected. An estimated **3,200 fraudulent listings per day** are slipping through the detection system.

### Detection

The `DriftDetector` class (`src/monitoring/drift.py`) monitors a sliding window of the last 1,000 predictions and compares current statistics against a saved baseline.

#### Step 1 — Probability Distribution Shift

```
Baseline mean probability:     0.72
Current mean probability:      0.54
Probability drift:             0.18  (threshold: 0.15) ⚠️ EXCEEDED
```

#### Step 2 — Increased Low-Confidence Predictions

```
Baseline low confidence ratio: 0.123
Current low confidence ratio:  0.348
Confidence drift:              0.225  (threshold: 0.15) ⚠️ EXCEEDED
```

#### Step 3 — Prediction Ratio Drift

```
Baseline AI prediction ratio:  0.48
Current AI prediction ratio:   0.26
Ratio shift:                   0.22  (threshold: 0.20) ⚠️ EXCEEDED
```

#### Alert Timeline

| Day | Event |
|-----|-------|
| Day 0 | New-generation AI images start appearing on platform |
| Day 3 | Low confidence ratio rises from 12% to 18% |
| Day 5 | Mean probability drops below 0.65 — first drift alert triggered |
| Day 7 | All three drift indicators exceed thresholds — **DRIFT_DETECTED** |
| Day 8 | On-call engineer investigates the alert |

### Root Cause Analysis

**The training dataset only contained images from older AI generators.** The newer generation produces images that lack the artifacts the model learned to detect:

- **Texture patterns**: Older generators produced subtle repetitive textures; newer models do not
- **Edge consistency**: Older generators had inconsistent edges; newer models handle this correctly
- **Color space distribution**: The color histogram of newer AI images is closer to real photos
- **Frequency domain**: High-frequency noise patterns used by the model are absent in newer generators

**Classification:** Data drift (covariate shift) — Severity P2

### Remediation

**Immediate (Day 8-9):**
1. Lowered decision threshold from 0.5 to 0.35
2. Enabled enhanced logging for low-confidence predictions
3. Notified platform moderation team

**Short-Term (Day 9-15):**
1. Collected 5,000 new AI-generated images from latest generators
2. Augmented training dataset, maintaining class balance
3. Retrained model — accuracy recovered to **87.4%**

**Deployment (Day 15-17):**
1. Canary deployment to 10% traffic for 48h
2. Full rollout after validation
3. Updated drift baseline

### Results

| Metric | Before Fix | After Fix |
|--------|-----------|-----------|
| **Accuracy** | 65.1% | 87.4% |
| **Precision (AI class)** | 58.3% | 88.1% |
| **Recall (AI class)** | 71.2% | 85.6% |
| **Low Confidence Ratio** | 34.8% | 9.7% |

### Prevention Measures

1. **Automated monitoring alerts** for drift score > 0.10 (warning) and > 0.15 (critical)
2. **Monthly retraining pipeline** with latest AI-generated samples
3. **AI generator registry** tracking new releases vs training data coverage
4. **Synthetic drift testing** to verify detector functionality

---

## Post-Mortem Template

Use this template for all P1 and P2 incidents. P3 incidents should use a simplified version.

```markdown
# Post-Mortem: [Incident Title]

## Metadata
- **Date:** YYYY-MM-DD
- **Duration:** HH:MM (from detection to resolution)
- **Severity:** P1 / P2 / P3
- **Author:** [Name]
- **Reviewers:** [Names]

## Summary
[1-2 sentence summary of what happened and the impact]

## Timeline (all times UTC)
| Time | Event |
|------|-------|
| HH:MM | Alert fired: [alert name] |
| HH:MM | On-call acknowledged |
| HH:MM | Root cause identified |
| HH:MM | Mitigation applied |
| HH:MM | Service restored |
| HH:MM | Incident closed |

## Impact
- **Duration of impact:** X minutes/hours
- **Users affected:** X% / all
- **Requests affected:** X failed out of Y total
- **Revenue impact:** $X (if applicable)
- **Data loss:** None / Description

## Root Cause
[Detailed technical explanation of why the incident occurred]

## Detection
- How was the incident detected? (alert / user report / manual observation)
- Was detection timely? If not, why?
- Time from incident start to detection: X minutes

## Mitigation
[What was done to restore service]

## Resolution
[What was done to fix the underlying issue]

## What Went Well
- [Thing 1]
- [Thing 2]

## What Went Wrong
- [Thing 1]
- [Thing 2]

## Where We Got Lucky
- [Thing 1]

## Action Items
| Priority | Action | Owner | Due Date | Status |
|----------|--------|-------|----------|--------|
| P1 | [Action] | [Name] | YYYY-MM-DD | Open |
| P2 | [Action] | [Name] | YYYY-MM-DD | Open |

## Lessons Learned
[Key takeaways that should inform future work]
```

---

*Document maintained as part of the MLOps incident response framework.*
*AI Product Photo Detector — M2 MLOps Project, JUNIA 2026*
