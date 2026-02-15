# Cost Management & Estimation

## Budget Configuration

| Setting | Value |
|---------|-------|
| **Monthly Budget** | 5.00 EUR |
| **Alert Thresholds** | 50%, 80%, 100% |
| **Billing Account** | Configured via GCP Console |
| **Budget ID** | Managed by Terraform |
| **Scope** | Single GCP project |

> Budget alerts are sent to the billing account administrators via email.
> To add specific notification channels (e.g., Pub/Sub, Slack), configure them in the
> GCP Console under Billing > Budgets & alerts.

### Manual Budget Setup (Console)

If the `gcloud` budget command fails (permission issues):

1. Go to **GCP Console → Billing → Budgets & alerts**
2. Click **Create Budget**
3. Set **Name**: `ai-product-detector-monthly`
4. **Scope** → Select your project
5. **Amount** → 5.00 EUR, **Budget type**: Specified amount
6. **Thresholds** → Add rules at 50%, 80%, 100% of actual spend
7. Under **Notifications**, add your email via a Monitoring Notification Channel
8. Click **Finish**

---

## Cloud Run Configuration

| Parameter | Value | Cost Impact |
|-----------|-------|-------------|
| **Max instances** | 3 | Caps parallel compute |
| **Min instances** | 0 | Scale to zero = no idle cost |
| **CPU** | 1 vCPU | Allocated per request |
| **Memory** | 2 GiB | Per container instance |
| **Concurrency** | 80 | Requests per instance |
| **Timeout** | 300s | Max request duration |
| **CPU throttling** | Disabled | CPU only during requests |
| **Startup CPU boost** | Enabled | Faster cold starts |

### Cloud Run Pricing (europe-west1)

| Resource | Price | Free Tier (monthly) |
|----------|-------|---------------------|
| CPU | $0.00002400/vCPU-second | 180,000 vCPU-seconds |
| Memory | $0.00000250/GiB-second | 360,000 GiB-seconds |
| Requests | $0.40/million | 2 million requests |

**Free tier translates to:**
- ~50 hours of 1 vCPU compute
- ~100 hours of 2 GiB memory (but only 50h if CPU-bound)
- 2 million requests

---

## Google Cloud Storage (GCS)

### Lifecycle Rules (configured)

| Rule | Condition | Action |
|------|-----------|--------|
| Temp file cleanup | Age > 90 days, prefix `tmp/`, `temp/`, `cache/` | Delete |
| Old versions cleanup | Non-current version > 30 days | Delete |

### GCS Pricing (europe-west1, Standard)

| Resource | Price | Free Tier (monthly) |
|----------|-------|---------------------|
| Storage | $0.020/GB/month | 5 GB |
| Class A ops (write) | $0.005/1,000 | 5,000 ops |
| Class B ops (read) | $0.0004/1,000 | 50,000 ops |
| Egress (same region) | Free | — |

**Estimated usage:** < 1 GB DVC data → **$0.00** (free tier).

---

## Artifact Registry

### Artifact Registry Pricing

| Resource | Price | Free Tier (monthly) |
|----------|-------|---------------------|
| Storage | $0.10/GB/month | 0.5 GB |
| Egress (same region) | Free | — |

**Estimated usage:** ~1-2 GB Docker image → **$0.05-0.15/month**.

---

## Monthly Cost Estimation

| Service | Estimated Usage | Estimated Cost |
|---------|-----------------|----------------|
| **Cloud Run** | Light usage (~100 req/day) | **$0.00** (free tier) |
| **GCS** | < 1 GB storage | **$0.00** (free tier) |
| **Artifact Registry** | ~1.5 GB (1 image) | **~$0.10** |
| **Cloud Build** (if used) | < 120 min/day | **$0.00** (free tier) |
| **Networking** | Same-region traffic | **$0.00** |
| | | |
| **Total estimated** | | **< $0.50/month** |

### GCP Free Tier Summary (Always Free)

| Service | Free Allowance |
|---------|---------------|
| Cloud Run | 2M requests, 180K vCPU-s, 360K GiB-s |
| Cloud Storage | 5 GB Standard, 5K class A ops, 50K class B ops |
| Cloud Build | 120 build-min/day |
| Artifact Registry | 0.5 GB storage |
| Cloud Logging | 50 GB/month |
| Cloud Monitoring | Basic (free) |

> **For a project with light traffic, monthly costs should stay well under the 5€ budget.**
> The main potential cost driver is Artifact Registry storage if many large images accumulate.

---

## Cost Optimization Applied

1. **Budget alert** at 5€/month with 50%/80%/100% thresholds
2. **Cloud Run** scale-to-zero (min-instances=0), max 3 instances
3. **GCS lifecycle** auto-deletes temp files (90d) and old versions (30d)
4. **Artifact Registry** cleanup: remove unused images periodically

## Maintenance Checklist

- [ ] Monthly: Check billing dashboard for unexpected charges
- [ ] Monthly: Clean up unused Artifact Registry images
- [ ] Quarterly: Review Cloud Run scaling settings
- [ ] Review budget alerts if project scope changes
