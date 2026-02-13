# ---------------------------------------------------------------------------
# Remote state backend (GCS)
# ---------------------------------------------------------------------------
# To enable remote state:
#
# 1. Create the bucket manually (one-time setup):
#    gsutil mb -l europe-west1 gs://<PROJECT_ID>-tfstate
#    gsutil versioning set on gs://<PROJECT_ID>-tfstate
#
# 2. Uncomment the backend block in the environment's main.tf
#    (environments/dev/main.tf or environments/prod/main.tf)
#
# 3. Run: terraform init -migrate-state
#
# Each environment uses a separate state prefix to avoid conflicts:
#   - dev:  terraform/state/dev
#   - prod: terraform/state/prod
# ---------------------------------------------------------------------------
