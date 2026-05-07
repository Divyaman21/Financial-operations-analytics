# Troubleshooting Guide

## Common Issues & Solutions

### Installation Issues

#### Prophet fails to install
**Error:** `cmdstan` or C++ compilation errors.
```
Solution:
pip install pystan==2.19.1.1
pip install prophet
```
On macOS, ensure Xcode CLI tools are installed:
```bash
xcode-select --install
```

#### LightGBM build fails on macOS
```bash
brew install libomp
pip install lightgbm
```

#### `scipy` version conflict on Python 3.9
The `requirements.txt` pins `scipy==1.13.1` for Python 3.9 compatibility. If on 3.10+:
```bash
pip install scipy>=1.14
```

---

### Pipeline Issues

#### `ModuleNotFoundError: No module named 'fo_analytics'`
The package needs to be installed in editable mode:
```bash
pip install -e .
```

#### Pipeline runs but produces empty artifacts
Check that the `artifacts/` directory is writable:
```bash
ls -la artifacts/
```

#### `ValueError: Series too short for holdout`
The synthetic dataset is too small. Increase `n_customers` in `generate_orders()` or reduce `HOLDOUT_WEEKS` in `config.py`.

---

### API Issues

#### `404 Not Found` on root URL
Visit http://localhost:8000/docs for the Swagger UI, or http://localhost:8000/ for the root endpoint.

#### `ModuleNotFoundError: No module named 'api.routers'`
Run uvicorn from the project root:
```bash
cd "Financial Analytical System"
uvicorn api.main:app --reload
```

#### CORS errors from frontend
Add CORS middleware to `api/main.py`:
```python
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
```

---

### Docker Issues

#### Container exits immediately
Check logs:
```bash
docker logs financial-analytics
```
Common cause: missing `requirements.txt` in image (check `.dockerignore`).

#### Artifacts not appearing on host
Ensure volume mount is correct:
```bash
docker run -v $(pwd)/artifacts:/app/artifacts financial-analytics:latest
```

---

### Model Issues

#### `ValueError: Model 'X' not found in registry`
Run the pipeline at least once to populate the model registry:
```bash
python run_pipeline.py
```

#### SHAP import error
```bash
pip install shap
```

---

### Testing Issues

#### Tests fail with import errors
Ensure the project is installed:
```bash
pip install -e .
pip install pytest pytest-cov
```

#### Tests are slow
The integration test runs the full pipeline. Skip it for quick iterations:
```bash
pytest tests/ -v --ignore=tests/test_integration.py
```

## Getting Help
- Check the [Architecture Guide](ARCHITECTURE.md) for system design
- Check the [API Guide](API_GUIDE.md) for endpoint details
- Open an issue on GitHub for bugs
