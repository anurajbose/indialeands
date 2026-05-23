# Publishing IndiaLend to PyPI

This guide walks through publishing IndiaLend to PyPI (and TestPyPI first).

## Prerequisites

1. **PyPI accounts**
   - Create account at https://pypi.org/account/register/
   - Create account at https://test.pypi.org/account/register/ (for test uploads)
   - Enable 2FA on both accounts

2. **API tokens** (recommended over passwords)
   - PyPI: https://pypi.org/manage/account/token/ (scope: entire account, then scoped to `indialend` after first upload)
   - TestPyPI: https://test.pypi.org/manage/account/token/

3. **Store tokens** in `~/.pypirc`:
   ```ini
   [distutils]
   index-servers =
       pypi
       testpypi

   [pypi]
   username = __token__
   password = pypi-AgEIcHlwaS5vcmc...   # your PyPI token

   [testpypi]
   repository = https://test.pypi.org/legacy/
   username = __token__
   password = pypi-AgEIdGVzdC5weXBpL...  # your TestPyPI token
   ```
   Then: `chmod 600 ~/.pypirc`

4. **Required tools** (already installed in venv):
   ```bash
   pip install --upgrade build twine
   ```

## Pre-publish checklist

- [ ] Version number bumped in `pyproject.toml`
- [ ] `CHANGELOG.md` updated with release notes
- [ ] All tests pass: `python -m pytest tests/`
- [ ] README renders correctly on GitHub
- [ ] No uncommitted changes: `git status`
- [ ] Tagged the release: `git tag v1.0.0 && git push --tags`

## Build the package

```bash
# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Build wheel and source distribution
python -m build

# Verify outputs
ls -la dist/
# Should see:
#   indialend-1.0.0-py3-none-any.whl
#   indialend-1.0.0.tar.gz
```

## Validate the package

```bash
# Check metadata and rendering
twine check dist/*
# Both files should show: PASSED

# Test install in a clean environment
python3 -m venv /tmp/indialend_test
source /tmp/indialend_test/bin/activate
pip install dist/indialend-1.0.0-py3-none-any.whl
python -c "import indialend; print(indialend.__version__)"
deactivate
rm -rf /tmp/indialend_test
```

## Upload to TestPyPI (always first!)

```bash
# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Install from TestPyPI to verify
pip install \
    --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    indialend

python -c "import indialend; print(indialend.__version__)"
```

View the package at: https://test.pypi.org/project/indialend/

## Upload to PyPI (production)

Only after TestPyPI verification passes:

```bash
twine upload dist/*
```

View the published package at: https://pypi.org/project/indialend/

## Verify the published package

```bash
# Wait ~1 minute for PyPI CDN to sync
pip install indialend
python -c "import indialend; print(indialend.__version__)"

# With optional dependencies
pip install indialend[gpu]      # PyTorch for Metal GPU neural network
pip install indialend[lightgbm] # LightGBM support
pip install indialend[xgboost]  # XGBoost support
pip install indialend[all]      # Everything
```

## Release workflow (future releases)

For subsequent releases:

```bash
# 1. Bump version in pyproject.toml (e.g., 1.0.0 -> 1.0.1)
# 2. Update CHANGELOG.md
# 3. Commit and tag
git add pyproject.toml CHANGELOG.md
git commit -m "Release v1.0.1"
git tag v1.0.1
git push origin main --tags

# 4. Build
rm -rf dist/ build/ *.egg-info
python -m build

# 5. Test on TestPyPI
twine upload --repository testpypi dist/*

# 6. Publish to PyPI
twine upload dist/*
```

## Troubleshooting

### "File already exists" error
PyPI doesn't allow re-uploading the same version. Bump the version in `pyproject.toml`.

### "Invalid credentials"
Re-check your `~/.pypirc` tokens. Tokens start with `pypi-`.

### Package name already taken
The name `indialend` must be unique on PyPI. If taken, update `name = "..."` in `pyproject.toml`.

### Twine check fails
- `long_description` error: Ensure `README.md` uses valid CommonMark
- `license` error: `pyproject.toml` uses `license = { file = "LICENSE" }` pointing to a real file
- Missing classifiers: Use official trove classifiers from https://pypi.org/classifiers/

### ImportError after install
- Missing module: Check `[tool.setuptools.packages.find]` in `pyproject.toml`
- Optional dep error: NeuralCreditScorer requires `pip install indialend[gpu]`

## GitHub Actions (automated publishing)

For CI-driven releases, create `.github/workflows/publish.yml`:

```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  build-and-publish:
    runs-on: ubuntu-latest
    environment:
      name: pypi
      url: https://pypi.org/p/indialend
    permissions:
      id-token: write  # For trusted publishing
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install build tools
        run: pip install build twine
      - name: Build package
        run: python -m build
      - name: Check package
        run: twine check dist/*
      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
```

For trusted publishing setup: https://docs.pypi.org/trusted-publishers/

## Package size optimization

Current package sizes:
- Wheel: ~71 KB
- Sdist: ~70 KB

These are small because we exclude `venv/`, `models/`, `data/`, and tests via `MANIFEST.in`.

## Current package metadata

```
Name:           indialend
Version:        1.0.0
Summary:        Unified Credit Decision Engine for Indian NBFC Lending
Home-page:      https://github.com/indialend/indialend
License:        MIT
Python:         >= 3.9
Dependencies:   numpy, pandas, scikit-learn, scipy, pydantic, joblib, python-dateutil
Optional:       lightgbm, xgboost, torch (gpu), fastapi+uvicorn (api)
```
