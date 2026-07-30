@echo off
echo ========================================
echo Screenshot Capture Helper Script
echo ========================================
echo.
echo This script will help you capture required screenshots
echo.
echo STEP 1: GitHub Screenshots
echo Opening GitHub repository...
start https://github.com/rihanaa-m/credit-risk-model/tree/task-2
echo.
echo Please capture these screenshots:
echo 1. Repository overview (save as: screenshots/github_repository.png)
echo 2. Commit history (save as: screenshots/commit_history.png)
echo 3. File structure (save as: screenshots/file_structure.png)
echo.
echo Press any key when ready for STEP 2...
pause >nul
echo.
echo STEP 2: Code Screenshots
echo Please open these files in your IDE and capture screenshots:
echo 1. src/woe_iv.py - calculate_woe_iv function (save as: screenshots/woe_iv_implementation.png)
echo 2. src/proxy_target.py - calculate_rfm_score function (save as: screenshots/enhanced_rfm.png)
echo 3. src/api/main.py - predict_enhanced function (save as: screenshots/api_endpoints.png)
echo.
echo Press any key when ready for STEP 3...
pause >nul
echo.
echo STEP 3: API Screenshots
echo Starting API server...
echo.
echo Please run this command in a separate terminal:
echo cd credit-risk-model
echo python -m uvicorn src.api.main:app --reload
echo.
echo Then open: http://localhost:8000/docs
echo.
echo Capture these screenshots:
echo 1. API documentation (save as: screenshots/api_documentation.png)
echo 2. API response example (save as: screenshots/api_response.png)
echo.
echo Press any key to open API documentation...
pause >nul
start http://localhost:8000/docs
echo.
echo ========================================
echo Screenshot Capture Helper Complete
echo ========================================
echo.
echo Don't forget to:
echo 1. Capture all required screenshots
echo 2. Use the exact filenames specified
echo 3. Ensure screenshots are clear and readable
echo 4. Keep files under 2MB each
echo.
echo For detailed guidance, see SCREENSHOT_GUIDE.md
echo.
pause