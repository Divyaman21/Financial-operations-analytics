# MASTER PROMPT: Complete Financial Analytics System
## For AI Agent Implementation of Remaining Features

---

## EXECUTIVE CONTEXT

**Status:** The GitHub repository has completed all "Future Work" items from the March 2025 Progress Report. The following **8 critical production features** are still missing and need immediate implementation.

**Your Task:** Implement these 8 features in the order listed. Each is self-contained and adds substantial business value.

---

## FEATURE #1: COMPREHENSIVE TEST SUITE
**Priority:** CRITICAL | Effort: 3 days | Business Value: Quality Assurance

### Objective
Create pytest suite with 70%+ coverage across all modules.

### Requirements
```
tests/
├── test_data.py           # Data generation, shapes, NaN handling
├── test_forecasting.py    # All 4 models (HW, SARIMA, Prophet, ensemble)
├── test_churn.py          # All 5 classifiers (LR, RF, GB, XGB, LGBM)
├── test_survival.py       # KM curves, Cox PH
├── test_segmentation.py   # RFM + K-Means
├── test_profitability.py  # Margins, Monte Carlo
├── test_export.py         # CSV, Parquet, JSON outputs
└── test_integration.py    # End-to-end pipeline
```

### Deliverables
1. **Unit tests** for each module (min 50 tests)
   - Data pipeline: validate shapes, types, ranges
   - Forecasting: MAPE < 5%, error metrics calculated
   - Churn: AUC-ROC > 0.62, F1 scores logged
   - Survival: KM non-negative, Cox p-values valid
   - Exports: file existence, schema validation

2. **Integration test** 
   - Run full `run_pipeline.py` end-to-end
   - Verify all 8 output files/directories generated
   - Validate artifact directory structure

3. **Coverage report**
   - `pytest --cov=src/fo_analytics --cov-report=html`
   - Minimum 70% line coverage

4. **CI/CD Configuration** (`.github/workflows/test.yml`)
   ```yaml
   name: Test
   on: [push, pull_request]
   jobs:
     test:
       runs-on: ubuntu-latest
       strategy:
         matrix:
           python-version: ['3.9', '3.10', '3.11']
       steps:
         - uses: actions/checkout@v3
         - name: Install deps
           run: pip install -r requirements.txt pytest pytest-cov
         - name: Run tests
           run: pytest tests/ --cov=src/fo_analytics --cov-report=xml
         - name: Upload coverage
           uses: codecov/codecov-action@v3
   ```

5. **Pre-commit hooks** (`.pre-commit-config.yaml`)
   ```yaml
   repos:
     - repo: https://github.com/psf/black
       rev: 23.3.0
       hooks:
         - id: black
     - repo: https://github.com/PyCQA/isort
       hooks:
         - id: isort
     - repo: https://github.com/PyCQA/flake8
       hooks:
         - id: flake8
           args: [--max-line-length=100]
   ```

### Success Criteria
- ✅ 70%+ line coverage (confirmed by pytest output)
- ✅ All tests pass on Python 3.9, 3.10, 3.11
- ✅ CI/CD pipeline runs < 5 min
- ✅ No critical linting errors (flake8 clean)

---

## FEATURE #2: DOCKER CONTAINERIZATION & DEPLOYMENT
**Priority:** HIGH | Effort: 2 days | Business Value: Reproducibility + Cloud Ready

### Objective
Containerize application for reproducible local/cloud deployment.

### Deliverables

1. **Dockerfile**
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Run pipeline
ENTRYPOINT ["python", "run_pipeline.py"]
```

2. **docker-compose.yml**
```yaml
version: '3.8'

services:
  analytics:
    build: .
    container_name: financial-analytics
    volumes:
      - ./artifacts:/app/artifacts
      - ./data:/app/data
    environment:
      - PYTHONUNBUFFERED=1
      - RANDOM_SEED=42
    command: python run_pipeline.py

  api:
    build: .
    container_name: financial-api
    ports:
      - "8000:8000"
    volumes:
      - ./artifacts:/app/artifacts
    command: uvicorn api.main:app --host 0.0.0.0 --reload
    depends_on:
      - analytics
```

3. **.dockerignore**
```
.git
.gitignore
__pycache__
*.pyc
.pytest_cache
.venv
venv
notebooks
.ipynb_checkpoints
*.egg-info
dist/
build/
.env
.DS_Store
```

4. **K8s manifests** (optional, `k8s/`)
   - `analytics-deployment.yaml`: pipeline job
   - `api-deployment.yaml`: FastAPI service
   - `configmap.yaml`: env variables
   - `pvc.yaml`: persistent volume for artifacts

5. **Documentation** (`docs/DOCKER_SETUP.md`)
```markdown
## Quick Start

### Build
\`\`\`bash
docker build -t financial-analytics:latest .
\`\`\`

### Run Analytics Pipeline
\`\`\`bash
docker run -v $(pwd)/artifacts:/app/artifacts financial-analytics:latest
\`\`\`

### Run API Server
\`\`\`bash
docker-compose up api
# Visit http://localhost:8000/docs
\`\`\`

### Production Deployment
\`\`\`bash
docker push financial-analytics:latest  # to registry
docker run --detach --restart unless-stopped \
  -v /data/artifacts:/app/artifacts \
  financial-analytics:latest
\`\`\`
```

### Success Criteria
- ✅ Docker image builds without errors
- ✅ Pipeline runs to completion inside container
- ✅ Artifacts mounted and accessible on host
- ✅ API server responds at http://localhost:8000/docs
- ✅ docker-compose up succeeds without warnings

---

## FEATURE #3: MODEL REGISTRY & VERSIONING
**Priority:** HIGH | Effort: 2 days | Business Value: MLOps + Reproducibility

### Objective
Serialize, version, and track all trained models.

### Deliverables

1. **Model Serializer** (`src/fo_analytics/registry/serializer.py`)
```python
import pickle
import json
from pathlib import Path
from datetime import datetime
import hashlib

class ModelSerializer:
    def __init__(self, artifact_dir='artifacts/models'):
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
    
    def save_model(self, model, name, model_type, metrics=None, 
                   hyperparams=None, dataset_hash=None):
        """
        Save model with metadata.
        
        Args:
            model: trained sklearn/xgb/prophet object
            name: model identifier (e.g., 'churn_xgboost')
            model_type: 'logistic_regression', 'xgboost', 'prophet', etc.
            metrics: dict of evaluation metrics
            hyperparams: dict of model hyperparameters
            dataset_hash: MD5 of training data for reproducibility
        
        Returns:
            version_id: timestamp-based version identifier
        """
        version_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = self.artifact_dir / f'{name}_{version_id}.pkl'
        
        # Serialize model
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        
        # Save metadata
        metadata = {
            'name': name,
            'type': model_type,
            'version': version_id,
            'timestamp': datetime.now().isoformat(),
            'filepath': str(filepath),
            'metrics': metrics or {},
            'hyperparams': hyperparams or {},
            'dataset_hash': dataset_hash,
            'file_size_mb': filepath.stat().st_size / (1024**2)
        }
        
        return version_id, filepath

class ModelRegistry:
    def __init__(self, manifest_file='artifacts/model_manifest.json'):
        self.manifest_file = Path(manifest_file)
        self.models = self._load_manifest()
    
    def register(self, name, filepath, version, metrics, hyperparams, 
                 model_type, dataset_hash):
        """Register model in manifest."""
        if name not in self.models:
            self.models[name] = {'versions': {}}
        
        self.models[name]['versions'][version] = {
            'filepath': str(filepath),
            'metrics': metrics,
            'hyperparams': hyperparams,
            'model_type': model_type,
            'dataset_hash': dataset_hash,
            'timestamp': datetime.now().isoformat()
        }
        
        self._save_manifest()
    
    def load(self, name, version='latest'):
        """Load model by name and version."""
        if name not in self.models:
            raise ValueError(f"Model '{name}' not found in registry")
        
        versions = self.models[name]['versions']
        if version == 'latest':
            version = max(versions.keys())
        
        if version not in versions:
            raise ValueError(f"Version '{version}' not found")
        
        filepath = versions[version]['filepath']
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def get_metrics(self, name, version='latest'):
        """Retrieve model metrics."""
        if version == 'latest':
            version = max(self.models[name]['versions'].keys())
        return self.models[name]['versions'][version]['metrics']
    
    def list_versions(self, name):
        """List all versions of a model."""
        return sorted(self.models[name]['versions'].keys())
    
    def _load_manifest(self):
        if self.manifest_file.exists():
            with open(self.manifest_file) as f:
                return json.load(f)
        return {}
    
    def _save_manifest(self):
        with open(self.manifest_file, 'w') as f:
            json.dump(self.models, f, indent=2)
```

2. **Integrate into Pipeline** (update `run_pipeline.py`)
```python
from fo_analytics.registry import ModelSerializer, ModelRegistry
import hashlib

# After training each model:
serializer = ModelSerializer()
registry = ModelRegistry()

# Example: Save XGBoost churn model
version_id, filepath = serializer.save_model(
    model=xgb_model,
    name='churn_xgboost',
    model_type='xgboost',
    metrics={'auc': 0.73, 'f1': 0.68, 'precision': 0.70},
    hyperparams={'n_estimators': 150, 'max_depth': 6, 'learning_rate': 0.1},
    dataset_hash=hashlib.md5(X_train.to_json().encode()).hexdigest()
)

registry.register(
    name='churn_xgboost',
    filepath=filepath,
    version=version_id,
    metrics={'auc': 0.73, 'f1': 0.68},
    hyperparams={'n_estimators': 150},
    model_type='xgboost',
    dataset_hash=hashlib.md5(X_train.to_json().encode()).hexdigest()
)
```

3. **Model Card Updates** (`docs/model_cards/`)
   - Add to each card:
     - Latest version ID
     - Serialization path
     - Data lineage (dataset hash)
     - Training reproducibility info

### Success Criteria
- ✅ All trained models serialized to `artifacts/models/`
- ✅ Model manifest JSON created and updated
- ✅ Can load any historical model: `registry.load('churn_xgboost', version='v20260507_143022')`
- ✅ Metrics and hyperparams stored alongside model
- ✅ Model cards reference version info

---

## FEATURE #4: DATA DRIFT MONITORING
**Priority:** MEDIUM | Effort: 2 days | Business Value: Production Safety

### Objective
Detect statistical distribution shifts in features and predictions.

### Deliverables

1. **Drift Detector** (`src/fo_analytics/monitoring/drift_detector.py`)
```python
import numpy as np
from scipy.stats import ks_2samp, chi2_contingency
import json

class DriftDetector:
    def __init__(self, baseline_file='artifacts/monitoring/baseline.json'):
        self.baseline = self._load_baseline(baseline_file)
    
    def compute_baseline(self, data, features):
        """Compute baseline statistics from training data."""
        baseline = {}
        for col in features:
            if data[col].dtype in ['float64', 'int64']:
                baseline[col] = {
                    'type': 'continuous',
                    'mean': float(data[col].mean()),
                    'std': float(data[col].std()),
                    'min': float(data[col].min()),
                    'max': float(data[col].max()),
                    'quantiles': {
                        'q25': float(data[col].quantile(0.25)),
                        'q50': float(data[col].quantile(0.50)),
                        'q75': float(data[col].quantile(0.75))
                    }
                }
            else:
                baseline[col] = {
                    'type': 'categorical',
                    'value_counts': data[col].value_counts().to_dict()
                }
        return baseline
    
    def check_drift(self, new_data, feature):
        """
        Detect drift using KS test (continuous) or Chi-squared (categorical).
        Returns: p_value, psi, alert_level
        """
        if feature not in self.baseline:
            return None
        
        baseline_info = self.baseline[feature]
        
        if baseline_info['type'] == 'continuous':
            # Kolmogorov-Smirnov test
            baseline_vals = np.random.normal(
                baseline_info['mean'], 
                baseline_info['std'], 
                10000
            )
            statistic, p_value = ks_2samp(baseline_vals, new_data[feature].dropna())
            
            # Population Stability Index
            psi = self._calculate_psi(baseline_vals, new_data[feature].dropna())
            
            if psi > 0.25:
                alert_level = 'CRITICAL'
            elif psi > 0.15:
                alert_level = 'WARNING'
            elif psi > 0.05:
                alert_level = 'CAUTION'
            else:
                alert_level = 'OK'
            
            return {
                'feature': feature,
                'test': 'Kolmogorov-Smirnov',
                'p_value': p_value,
                'psi': psi,
                'alert_level': alert_level
            }
        
        return None
    
    def _calculate_psi(self, baseline, current, bins=10):
        """Calculate Population Stability Index."""
        baseline_counts = np.histogram(baseline, bins=bins)[0]
        current_counts = np.histogram(current, bins=bins)[0]
        
        baseline_pct = baseline_counts / baseline_counts.sum()
        current_pct = current_counts / current_counts.sum()
        
        psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
        return psi
    
    def check_all(self, new_data):
        """Check drift for all features."""
        results = []
        for feature in self.baseline.keys():
            if feature in new_data.columns:
                drift_result = self.check_drift(new_data, feature)
                if drift_result:
                    results.append(drift_result)
        
        max_psi = max([r['psi'] for r in results], default=0)
        alert_severity = max([r['alert_level'] for r in results], default='OK')
        
        return {
            'results': results,
            'max_psi': max_psi,
            'alert_severity': alert_severity,
            'timestamp': datetime.now().isoformat()
        }
    
    def _load_baseline(self, filepath):
        if Path(filepath).exists():
            with open(filepath) as f:
                return json.load(f)
        return {}
```

2. **Generate Drift Report** (`src/fo_analytics/monitoring/reporter.py`)
```python
def generate_drift_report(drift_results, output_file='artifacts/monitoring/drift_report.html'):
    """Generate HTML drift report with visualizations."""
    html = """
    <html>
    <head>
        <title>Data Drift Report</title>
        <style>
            body { font-family: Arial; margin: 20px; }
            .warning { color: red; font-weight: bold; }
            .caution { color: orange; font-weight: bold; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #4CAF50; color: white; }
        </style>
    </head>
    <body>
        <h1>Data Drift Analysis Report</h1>
        <p>Generated: {timestamp}</p>
        <p>Overall Alert Level: <span class="{alert_class}">{alert_level}</span></p>
        <table>
            <tr>
                <th>Feature</th>
                <th>Test</th>
                <th>P-Value</th>
                <th>PSI</th>
                <th>Alert Level</th>
            </tr>
            {rows}
        </table>
    </body>
    </html>
    """
    # Generate rows from results...
    # Save to HTML file
```

3. **Integration in Pipeline** (update `run_pipeline.py`)
```python
from fo_analytics.monitoring import DriftDetector

# After loading new data:
detector = DriftDetector('artifacts/monitoring/baseline.json')
drift_results = detector.check_all(new_customer_data)

if drift_results['alert_severity'] in ['WARNING', 'CRITICAL']:
    logger.warning(f"⚠️ Data drift detected: {drift_results['max_psi']:.3f} PSI")
    # Optional: trigger retraining
```

### Success Criteria
- ✅ Baseline computed from training data → `artifacts/monitoring/baseline.json`
- ✅ Drift detected on new data with KS test + PSI
- ✅ HTML report generated → `artifacts/monitoring/drift_report.html`
- ✅ Warnings logged when PSI > 0.15
- ✅ No false alarms on in-distribution data

---

## FEATURE #5: EXPLAINABILITY (SHAP VALUES)
**Priority:** MEDIUM | Effort: 2 days | Business Value: Model Trust + Compliance

### Objective
Add SHAP/LIME explanations to churn models.

### Deliverables

1. **SHAP Integration** (`src/fo_analytics/explainability/shap_explainer.py`)
```python
import shap
import matplotlib.pyplot as plt

class SHAPExplainer:
    def __init__(self, model, X_train):
        self.model = model
        self.explainer = shap.TreeExplainer(model)  # For tree models
        self.shap_values = self.explainer.shap_values(X_train)
    
    def summary_plot(self, X_test, output_file=None):
        """Generate SHAP summary plot."""
        shap_vals_test = self.explainer.shap_values(X_test)
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_vals_test, X_test, plot_type="bar", show=False)
        
        if output_file:
            plt.savefig(output_file, dpi=100, bbox_inches='tight')
        plt.close()
    
    def force_plot(self, X_sample, output_file=None):
        """Generate SHAP force plot for single prediction."""
        shap_val = self.explainer.shap_values(X_sample)[0]
        
        plt.figure(figsize=(12, 3))
        shap.force_plot(self.explainer.expected_value, shap_val, X_sample, 
                       plot_type="matplotlib", show=False)
        
        if output_file:
            plt.savefig(output_file, dpi=100, bbox_inches='tight')
        plt.close()
    
    def explain_instance(self, X_instance):
        """Return top 5 driving features for a single prediction."""
        shap_val = self.explainer.shap_values(X_instance)[0]
        
        feature_importance = list(zip(X_instance.columns, shap_val))
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        
        return {
            'top_features': [f[0] for f in feature_importance[:5]],
            'contributions': [float(f[1]) for f in feature_importance[:5]]
        }
```

2. **Dashboard Integration**
   - Add "Explainability" tab to executive dashboard
   - Embed SHAP summary plots
   - Link to sample explanations

3. **Model Card Updates**
   - Include top 5 SHAP features per model
   - Add sample prediction explanation

### Success Criteria
- ✅ SHAP values computed for all tree models
- ✅ Summary plots saved to `artifacts/explainability/`
- ✅ Sample explanations generated
- ✅ Dashboard includes explainability section
- ✅ Feature importance comparison documented

---

## FEATURE #6: FORECASTING ENSEMBLE
**Priority:** MEDIUM | Effort: 2 days | Business Value: +2–3% Accuracy

### Objective
Combine HW, SARIMA, Prophet via ensemble.

### Deliverables

1. **Ensemble Aggregator** (`src/fo_analytics/forecasting/ensemble.py`)
```python
import numpy as np
from sklearn.linear_model import LinearRegression

class ForecastEnsemble:
    def __init__(self, models_dict):
        """
        models_dict: {
            'holt_winters': hw_fitted,
            'sarima': sarima_fitted,
            'prophet': prophet_fitted
        }
        """
        self.models = models_dict
        self.weights = None
    
    def fit_weights(self, y_true, forecasts_dict, method='inverse_variance'):
        """
        Compute ensemble weights from holdout performance.
        
        method: 'equal', 'inverse_variance', 'stacking'
        """
        if method == 'inverse_variance':
            rmses = {
                name: np.sqrt(np.mean((y_true - forecast)**2))
                for name, forecast in forecasts_dict.items()
            }
            weights = {name: 1/rmse**2 for name, rmse in rmses.items()}
            total = sum(weights.values())
            self.weights = {name: w/total for name, w in weights.items()}
        
        elif method == 'stacking':
            X_meta = np.column_stack([
                forecasts_dict[name] for name in sorted(forecasts_dict.keys())
            ])
            meta_model = LinearRegression()
            meta_model.fit(X_meta, y_true)
            self.weights = {
                name: float(coef)
                for name, coef in zip(sorted(forecasts_dict.keys()), 
                                     meta_model.coef_)
            }
        
        return self.weights
    
    def predict(self, h=26, forecasts_dict=None):
        """Generate ensemble forecast."""
        if self.weights is None:
            self.weights = {name: 1/len(self.models) 
                           for name in self.models.keys()}
        
        if forecasts_dict is None:
            forecasts_dict = {
                name: model.forecast(h)
                for name, model in self.models.items()
            }
        
        ensemble_forecast = np.zeros(h)
        for name, forecast in forecasts_dict.items():
            weight = self.weights.get(name, 0)
            ensemble_forecast += weight * forecast
        
        return ensemble_forecast
```

2. **Backtesting**
   - Train on first 130 weeks, test on last 26
   - Compare individual models vs. ensemble
   - Report MAPE improvement

3. **Update Dashboard**
   - Add ensemble forecast to chart
   - Show individual model predictions for comparison
   - Display MAPE scores per model

### Success Criteria
- ✅ Ensemble MAPE < min(individual model MAPEs)
- ✅ Weights are data-driven
- ✅ Dashboard shows all 4 forecasts + ensemble
- ✅ Backtest documented

---

## FEATURE #7: AUTOMATED ALERTING SYSTEM
**Priority:** MEDIUM | Effort: 1 day | Business Value: Actionability

### Objective
Generate real-time alerts on metric deviations.

### Deliverables

1. **Alert Engine** (`src/fo_analytics/alerting/alert_engine.py`)
```python
class AlertEngine:
    def __init__(self, config_file='config/alerts.yaml'):
        self.config = self._load_config(config_file)
        self.alerts = []
    
    def check_churn_spike(self, current_rate, historical_mean, historical_std):
        """Alert if churn_rate > mean + 3*sigma."""
        threshold = historical_mean + (3 * historical_std)
        if current_rate > threshold:
            self.alerts.append({
                'rule': 'churn_spike',
                'severity': 'WARNING',
                'message': f'Churn rate {current_rate:.1%} exceeds threshold {threshold:.1%}',
                'value': current_rate,
                'threshold': threshold,
                'timestamp': datetime.now().isoformat()
            })
    
    def check_margin_threshold(self, current_margin, min_margin=-0.01):
        """Alert if margin < threshold."""
        if current_margin < min_margin:
            self.alerts.append({
                'rule': 'margin_decline',
                'severity': 'CRITICAL',
                'message': f'Net margin {current_margin:.2%} below minimum {min_margin:.2%}',
                'value': current_margin,
                'threshold': min_margin,
                'timestamp': datetime.now().isoformat()
            })
    
    def check_forecast_error(self, mape, threshold=0.05):
        """Alert if forecast error exceeds threshold."""
        if mape > threshold:
            self.alerts.append({
                'rule': 'forecast_error',
                'severity': 'CAUTION',
                'message': f'Forecast MAPE {mape:.2%} exceeds {threshold:.2%}',
                'value': mape,
                'threshold': threshold,
                'timestamp': datetime.now().isoformat()
            })
    
    def check_data_drift(self, max_psi, threshold=0.15):
        """Alert if data drift detected."""
        if max_psi > threshold:
            self.alerts.append({
                'rule': 'data_drift',
                'severity': 'ALERT',
                'message': f'Data drift PSI {max_psi:.3f} exceeds {threshold:.3f}',
                'value': max_psi,
                'threshold': threshold,
                'timestamp': datetime.now().isoformat()
            })
    
    def generate_report(self, output_file='artifacts/alerts.html'):
        """Generate HTML alert report."""
        severity_colors = {'CRITICAL': 'red', 'WARNING': 'orange', 
                          'ALERT': 'yellow', 'INFO': 'blue'}
        
        html = f"""
        <html>
        <head>
            <title>Alerts Report</title>
            <style>
                body {{ font-family: Arial; margin: 20px; }}
                .alert {{ margin: 10px 0; padding: 10px; border-left: 4px solid; }}
                .CRITICAL {{ border-color: red; background: #ffe6e6; }}
                .WARNING {{ border-color: orange; background: #fff3e0; }}
            </style>
        </head>
        <body>
            <h1>🚨 Alerts Report</h1>
            <p>Generated: {datetime.now().isoformat()}</p>
            <p>Total Alerts: {len(self.alerts)}</p>
            {"".join([
                f'<div class="alert {a["severity"]}">'
                f'<strong>[{a["severity"]}]</strong> {a["message"]}</div>'
                for a in self.alerts
            ])}
        </body>
        </html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html)
```

2. **Email Alerts** (optional)
```python
def send_email_alert(alerts, recipients, smtp_config):
    """Send email for CRITICAL alerts."""
    critical = [a for a in alerts if a['severity'] == 'CRITICAL']
    if critical:
        # Use smtplib to send notification
        pass
```

### Success Criteria
- ✅ Rules configurable in YAML
- ✅ Alerts generated and logged
- ✅ HTML report created
- ✅ No false positives on stable data

---

## FEATURE #8: COMPREHENSIVE DOCUMENTATION & API
**Priority:** HIGH | Effort: 2 days | Business Value: Enterprise Adoption

### Objective
Complete documentation and REST API for production serving.

### Deliverables

1. **FastAPI Endpoints** (`api/routers/`)
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Financial Analytics API")

class ChurnPredictionRequest(BaseModel):
    customer_id: int
    monthly_charges: float
    tenure_days: int
    support_tickets: int
    payment_delays: int
    login_frequency: int
    contract_type: str
    region: str

@app.post("/api/v1/churn/predict")
async def predict_churn(request: ChurnPredictionRequest):
    """Predict churn probability for a customer."""
    # Load best model from registry
    model = registry.load('churn_xgboost', version='latest')
    
    # Prepare features
    features = prepare_features(request.dict())
    
    # Get prediction
    prob = model.predict_proba([features])[0, 1]
    
    return {
        "customer_id": request.customer_id,
        "churn_probability": float(prob),
        "risk_tier": "High" if prob > 0.6 else "Medium" if prob > 0.3 else "Low",
        "clv": estimate_clv(request, prob),
        "model_version": "latest"
    }

@app.get("/api/v1/forecast/26-weeks")
async def forecast_revenue(scenario: str = "base"):
    """Get 26-week revenue forecast."""
    model = registry.load('forecast_ensemble', version='latest')
    
    forecast = model.predict(h=26)
    if scenario == 'optimistic':
        forecast *= 1.08
    elif scenario == 'pessimistic':
        forecast *= 0.92
    
    return {
        "scenario": scenario,
        "forecast": forecast.tolist(),
        "model": "ensemble",
        "metrics": {"mape": 0.031, "rmse": 5500}
    }

@app.get("/api/v1/health")
async def health_check():
    """Check API and model health."""
    return {
        "status": "healthy",
        "models": {
            "churn": "ready",
            "forecast": "ready",
            "survival": "ready"
        },
        "last_update": datetime.now().isoformat()
    }
```

2. **Documentation** (`docs/`)
   - **API_GUIDE.md**: Endpoint reference with curl examples
   - **ARCHITECTURE.md**: System design + data flow
   - **TROUBLESHOOTING.md**: Common issues + solutions
   - **QUICK_START.md**: 5-minute setup guide

3. **Expanded Model Cards** (`docs/model_cards/`)
   - Churn: architecture, hyperparameters, SHAP insights
   - Forecast: ensemble weights, backtesting results
   - Survival: KM curves interpretation, Cox coefficients
   - Profitability: margin drivers, scenario results

4. **README Updates**
   - Quick start with Docker
   - API endpoint summary
   - FAQ section
   - Links to all guides

### Success Criteria
- ✅ 5+ API endpoints implemented
- ✅ OpenAPI/Swagger docs auto-generated
- ✅ All guides written (> 2,000 words each)
- ✅ Setup < 10 minutes
- ✅ API < 500ms per request

---

## IMPLEMENTATION ORDER

```
Week 1: Feature #1 (Testing) + #2 (Docker)           → 5 days
Week 2: Feature #3 (Registry) + #4 (Drift)           → 4 days
Week 3: Feature #5 (SHAP) + #6 (Ensemble)            → 4 days
Week 3–4: Feature #7 (Alerts) + #8 (Docs + API)      → 3 days

TOTAL: ~16 days (2–3 weeks full-time)
```

---

## SUCCESS METRICS

After completion, the system should have:

- ✅ 70%+ test coverage
- ✅ Docker builds successfully
- ✅ All models versioned and tracked
- ✅ Drift monitoring enabled
- ✅ SHAP explanations for churn models
- ✅ Forecast MAPE < 3.5% (ensemble)
- ✅ Automated alerts on metrics
- ✅ Production API with 6+ endpoints
- ✅ Complete documentation (25+ pages)
- ✅ Ready for cloud deployment

---

## FINAL CHECKLIST FOR AI AGENT

Before implementing each feature:
- [ ] Review existing code structure
- [ ] Check current dependencies in `requirements.txt`
- [ ] Run `python run_pipeline.py` to baseline
- [ ] Create feature branch: `git checkout -b feature/[name]`

During implementation:
- [ ] Write tests first (TDD)
- [ ] Maintain backward compatibility
- [ ] Add comprehensive docstrings
- [ ] Update README after completion

After each feature:
- [ ] All tests pass: `pytest tests/`
- [ ] Code formatted: `black src/`
- [ ] Linting clean: `flake8 src/`
- [ ] Git commit with clear message
- [ ] Verify artifact outputs

**Now implement these 8 features to complete the Financial Analytics system for enterprise production.**

