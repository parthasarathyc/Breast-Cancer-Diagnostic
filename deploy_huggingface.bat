@echo off
setlocal enabledelayedexpansion

echo ========================================================
echo   Breast CARE-MA - Hugging Face Spaces Deployment
echo ========================================================
echo.
echo Step 1: Ensure you have created a new Space on huggingface.co:
echo    - Go to: https://huggingface.co/new-space
echo    - Select Space SDK: Docker (Blank)
echo.

set /p HF_USERNAME="Enter your Hugging Face Username: "
set /p HF_SPACENAME="Enter your Space Name (e.g. breast-care-ma): "

if "%HF_USERNAME%"=="" (
    echo [!] Username cannot be empty. Exiting.
    pause
    exit /b 1
)

if "%HF_SPACENAME%"=="" (
    set HF_SPACENAME=breast-care-ma
)

set REMOTE_URL=https://huggingface.co/spaces/%HF_USERNAME%/%HF_SPACENAME%

echo.
echo [*] Target Space: %REMOTE_URL%
echo.

:: Initialize git if not already initialized
if not exist .git (
    echo [*] Initializing Git repository...
    git init
    git branch -M main
)

:: Check if remote already exists
git remote remove space 2>nul
git remote add space %REMOTE_URL%

echo [*] Adding project files to Git...
git add Dockerfile .dockerignore README.md backend frontend

echo [*] Committing changes...
git commit -m "Deploy Breast CARE-MA to Hugging Face Spaces" 2>nul

echo.
echo [*] Pushing to Hugging Face Space...
echo (If prompted, enter your Hugging Face Username and Access Token as Password)
echo.
git push space main --force

echo.
echo ========================================================
echo [SUCCESS] Push completed!
echo Visit your live Space at:
echo https://huggingface.co/spaces/%HF_USERNAME%/%HF_SPACENAME%
echo ========================================================
pause
