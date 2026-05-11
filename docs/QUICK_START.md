# Quick Start Guide

Get the Financial Operations Analytics System running in under 10 minutes.

## Prerequisites
- Python 3.9, 3.10, or 3.11
- pip (latest)
- Git

## 1. Clone the Repository
```bash
git clone https://github.com/Divyaman21/Financial-operations-analytics.git
cd Financial-operations-analytics
```

## 2. Create a Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip3 install -U pip
```

## 3. Install Dependencies
```bash
pip3 install -r requirements.txt
pip3 install -e ".[api]"
```

## 4. Run the Pipeline
```bash
python3 run_pipeline.py
```

This generates all analytics and saves outputs to `artifacts/`:
- **Dashboard**: `artifacts/dashboard/index.html`
- **BI exports**: `artifacts/bi_export/` (CSV + Parquet)
- **Figures**: `artifacts/figures/` (plots and diagnostics)
- **Metrics**: `artifacts/metrics/` (JSON logs)
- **Models**: `artifacts/models/` (serialized model files)
- **Alerts**: `artifacts/alerts.html`
- **Drift report**: `artifacts/monitoring/drift_report.html`

## 5. View the Dashboard
```bash
open artifacts/dashboard/index.html   # macOS
xdg-open artifacts/dashboard/index.html  # Linux
```

## 6. Start the API (Optional)
```bash
uvicorn api.main:app --reload
```
Visit **http://localhost:8000/docs** for interactive API documentation.

## 7. Run with Docker (Alternative)
```bash
docker build -t financial-analytics:latest .
docker run -v $(pwd)/artifacts:/app/artifacts financial-analytics:latest
```

## 8. Run Tests
```bash
pip3 install pytest pytest-cov
PYTHONPATH=src python3 -m pytest tests/ -v --cov=src/fo_analytics
```

## Next Steps
- Read the [Architecture Guide](ARCHITECTURE.md) for system design
- Read the [API Guide](API_GUIDE.md) for endpoint reference
- Review [Model Cards](model_cards/) for model documentation
- Check [Docker Setup](DOCKER_SETUP.md) for containerised deployment
