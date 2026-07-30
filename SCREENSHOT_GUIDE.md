# Screenshot Capture Guide for Final Submission

## 📸 Required Screenshots Checklist

### 1. GitHub Repository Screenshots

#### Screenshot 1: Repository Overview
**URL**: https://github.com/rihanaa-m/credit-risk-model/tree/task-2

**What to Capture**:
- Repository name and description
- Branch selector showing "task-2"
- File tree structure
- Recent commit message

**How to Capture**:
1. Open the URL in your browser
2. Press `Windows + Shift + S` (Windows) or `Cmd + Shift + 4` (Mac)
3. Select the repository overview area
4. Save as: `screenshots/github_repository.png`

#### Screenshot 2: Commit History
**URL**: https://github.com/rihanaa-m/credit-risk-model/commits/task-2

**What to Capture**:
- Recent commits showing Week 12 enhancements
- Commit messages for the key changes
- Author and date information

**How to Capture**:
1. Navigate to the commits page
2. Screenshot the recent commits (last 5-10)
3. Save as: `screenshots/commit_history.png`

#### Screenshot 3: File Structure
**URL**: https://github.com/rihanaa-m/credit-risk-model/tree/task-2

**What to Capture**:
- Complete file tree
- New files highlighted (woe_iv.py, main_train_enhanced.py)
- Enhanced files (proxy_target.py, data_processing.py, etc.)

**How to Capture**:
1. Expand the file tree if needed
2. Screenshot the complete structure
3. Save as: `screenshots/file_structure.png`

---

### 2. Code Implementation Screenshots

#### Screenshot 4: WoE/IV Implementation
**File**: `src/woe_iv.py`

**What to Capture**:
- The calculate_woe_iv function
- WoETransformer class
- Key comments showing industry standards

**How to Capture**:
1. Open `src/woe_iv.py` in your IDE
2. Navigate to lines 50-100 (calculate_woe_iv function)
3. Screenshot with proper syntax highlighting
4. Save as: `screenshots/woe_iv_implementation.png`

#### Screenshot 5: Enhanced RFM Scoring
**File**: `src/proxy_target.py`

**What to Capture**:
- calculate_rfm_score function
- Business-aligned weights (0.4, 0.3, 0.3)
- Composite score calculation

**How to Capture**:
1. Open `src/proxy_target.py` in your IDE
2. Navigate to the calculate_rfm_score function (around line 115)
3. Screenshot the scoring logic
4. Save as: `screenshots/enhanced_rfm.png`

#### Screenshot 6: API Dual Endpoints
**File**: `src/api/main.py`

**What to Capture**:
- Both /predict and /predict-enhanced endpoints
- Response time tracking code
- Enhanced audit trail

**How to Capture**:
1. Open `src/api/main.py` in your IDE
2. Navigate to the predict_enhanced function (around line 140)
3. Screenshot the dual endpoint implementation
4. Save as: `screenshots/api_endpoints.png`

---

### 3. API Demonstration Screenshots

#### Screenshot 7: API Documentation
**URL**: http://localhost:8000/docs (after running API)

**What to Capture**:
- Swagger UI interface
- Both endpoints visible
- Request/response schemas

**How to Run API**:
```bash
cd credit-risk-model
python -m uvicorn src.api.main:app --reload
```

**How to Capture**:
1. Run the API command above
2. Open http://localhost:8000/docs in browser
3. Screenshot the full Swagger UI
4. Save as: `screenshots/api_documentation.png`

#### Screenshot 8: API Response Example
**URL**: http://localhost:8000/docs (after running API)

**What to Capture**:
- Try it out section for /predict-enhanced
- Example request body
- Response showing response_time_ms

**How to Capture**:
1. Click "Try it out" on /predict-enhanced
2. Execute with sample data
3. Screenshot the response
4. Save as: `screenshots/api_response.png`

---

### 4. MLflow Tracking Screenshots (Optional but Recommended)

#### Screenshot 9: MLflow Dashboard
**URL**: http://localhost:5000 (after running MLflow)

**What to Capture**:
- Experiment comparison
- Model metrics
- Run parameters

**How to Run MLflow**:
```bash
cd credit-risk-model
mlflow ui
```

**How to Capture**:
1. Run MLflow UI
2. Navigate to the credit_risk_week12_enhanced experiment
3. Screenshot the dashboard
4. Save as: `screenshots/mlflow_dashboard.png`

---

### 5. Before/After Comparison Screenshots

#### Screenshot 10: Code Comparison
**What to Capture**:
- Side-by-side comparison of original vs enhanced code
- Highlight key differences
- Add annotations

**How to Create**:
1. Use GitHub's compare view
2. Or create a manual comparison in your editor
3. Save as: `screenshots/before_after_comparison.png`

---

## 🎯 Screenshot Quality Guidelines

### Technical Requirements
- **Resolution**: Minimum 1280x720 pixels
- **Format**: PNG (preferred) or high-quality JPG
- **File Size**: Under 2MB per screenshot
- **Naming**: Use the exact filenames specified above

### Visual Guidelines
- **Clarity**: Text should be readable without zooming
- **Context**: Include relevant UI elements (browser tabs, IDE panels)
- **Consistency**: Use consistent screenshot style across all images
- **Labels**: Add arrows/annotations if helpful (optional)

### Content Guidelines
- **Completeness**: Show complete functions or relevant sections
- **Accuracy**: Ensure code is current and matches repository
- **Relevance**: Focus on Week 12 enhancements
- **Professional**: Avoid personal information or unrelated tabs

---

## 📋 Screenshot Capture Order

### Recommended Sequence
1. GitHub repository screenshots (can be done immediately)
2. Code implementation screenshots (open files in IDE)
3. API demonstration screenshots (requires running API)
4. MLflow screenshots (optional, requires running MLflow)
5. Before/after comparisons (can be done last)

### Time Estimate
- GitHub screenshots: 10 minutes
- Code screenshots: 15 minutes
- API screenshots: 20 minutes (includes running API)
- MLflow screenshots: 15 minutes (optional)
- Total: ~1 hour (or 45 minutes without MLflow)

---

## 🔧 Tools for Screenshot Capture

### Windows
- **Built-in**: Windows + Shift + S (Snipping Tool)
- **Lightshot**: https://app.prntscr.com/en/index.html
- **ShareX**: https://getsharex.com/

### Mac
- **Built-in**: Cmd + Shift + 4 (partial) or Cmd + Shift + 5 (full)
- **CleanShot X**: https://cleanshot.com/
- **Snagit**: https://www.techsmith.com/screen-capture.html

### Browser Extensions
- **Awesome Screenshot**: Chrome/Firefox extension
- **Full Page Screen Capture**: Chrome extension

---

## ✅ Final Checklist

### Before Submitting Screenshots
- [ ] All 8-10 required screenshots captured
- [ ] Filenames match the guide exactly
- [ ] All screenshots are clear and readable
- [ ] File sizes are under 2MB each
- [ ] Screenshots are in the `screenshots/` folder
- [ ] No personal information visible
- [ ] Code matches current repository state

### Screenshot Inventory
- [ ] github_repository.png
- [ ] commit_history.png
- [ ] file_structure.png
- [ ] woe_iv_implementation.png
- [ ] enhanced_rfm.png
- [ ] api_endpoints.png
- [ ] api_documentation.png
- [ ] api_response.png
- [ ] mlflow_dashboard.png (optional)
- [ ] before_after_comparison.png (optional)

---

## 🚀 Quick Start Commands

### GitHub Screenshots (Immediate)
```bash
# No commands needed - just open browser and navigate to:
# https://github.com/rihanaa-m/credit-risk-model/tree/task-2
```

### Code Screenshots (Immediate)
```bash
# Open files in your IDE:
# - src/woe_iv.py
# - src/proxy_target.py
# - src/api/main.py
```

### API Screenshots (Requires Running API)
```bash
cd credit-risk-model
python -m uvicorn src.api.main:app --reload
# Then open: http://localhost:8000/docs
```

### MLflow Screenshots (Optional)
```bash
cd credit-risk-model
mlflow ui
# Then open: http://localhost:5000
```

---

## 📞 Troubleshooting

### API Won't Start
**Issue**: Module not found errors
**Solution**: Ensure you're in the credit-risk-model directory and dependencies are installed

### MLflow Won't Start
**Issue**: Port already in use
**Solution**: Use `mlflow ui -p 5001` and navigate to localhost:5001

### Screenshots Too Large
**Issue**: File size exceeds 2MB
**Solution**: Compress images or use PNG compression tool

### Text Not Readable
**Issue**: Screenshot text is blurry
**Solution**: Increase resolution or zoom in before capturing

---

**Good luck with your screenshot capture! 📸**