# Privacy Policy — AI Product Photo Detector

**Last updated:** 2025-07-18

## Data Processing

This API classifies product images as **real** or **AI-generated**. Here is how we handle your data:

### Images

- **In-memory only** — Uploaded images are processed entirely in memory and **never saved to disk**.
- **No persistence** — Image bytes are read, transformed into a tensor, passed through the model, and immediately discarded.
- **No caching** — We do not cache uploaded images or their representations.
- **No third-party sharing** — Images are not sent to any external service.

### Logging

- We log **operational metadata only**: prediction result (real/ai_generated), probability score, inference time, and request status.
- We **never log** image content, filenames containing personal information, IP addresses, or any user-identifiable data in application logs.
- Structured logs are used for monitoring API health and performance.

### Prometheus Metrics

Exposed at `/metrics`, our Prometheus metrics contain only **aggregate counters and histograms**:
- Total predictions by status/result/confidence
- Latency distributions
- Error counts by type
- Batch sizes

**No personal data** (IPs, filenames, image content) appears in metrics.

### Rate Limiting

- IP-based rate limiting is used to prevent abuse (30 requests/minute for `/predict`).
- IP addresses are used **only in-memory** for rate limit tracking and are **not stored or logged**.

### Authentication

- When enabled, API key authentication validates requests.
- API keys are compared in-memory; failed authentication attempts are not logged with the key value.

## GDPR Compliance

Under the EU General Data Protection Regulation (GDPR):

- **No personal data is collected or stored** by this API.
- Images are processed as a **stateless service** — each request is independent.
- No cookies, sessions, or user tracking of any kind.
- No data retention — nothing persists beyond the request lifecycle.

If you believe your data has been processed inappropriately, contact the maintainer.

## Contact

**Nolan Cacheux** — [GitHub](https://github.com/nolancacheux)
