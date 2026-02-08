@echo off
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT=%SCRIPT_DIR%..\.."

set "PYTHON_EXE=%REPO_ROOT%\.pixi\envs\default\python.exe"
if not exist "%PYTHON_EXE%" (
  set "PYTHON_EXE=python"
)

set "EXTRACT_SCRIPT=%SCRIPT_DIR%extract_main_store.py"
set "CONTEXT_SCRIPT=%SCRIPT_DIR%context_builder.py"
set "TASK_SCRIPT=%SCRIPT_DIR%task_builder.py"
set "STORE_SCRIPT=%SCRIPT_DIR%task_store.py"

echo Using Python: %PYTHON_EXE%
echo Repo root: %REPO_ROOT%
echo.

for /L %%U in (1,1,10) do (
  echo [User %%U] extract states/events/logs
  "%PYTHON_EXE%" "%EXTRACT_SCRIPT%" --user-id %%U
  if errorlevel 1 goto :fail

  echo [User %%U] build raw
  "%PYTHON_EXE%" "%CONTEXT_SCRIPT%" --user-id %%U
  if errorlevel 1 goto :fail

  for %%T in (t1 t2 t3 t4 t5 t6) do (
    echo [User %%U] build %%T
    "%PYTHON_EXE%" "%TASK_SCRIPT%" --user-id %%U --tag %%T
    if errorlevel 1 goto :fail
  )

  echo [User %%U] build tasks.json
  "%PYTHON_EXE%" "%STORE_SCRIPT%" --user-id %%U
  if errorlevel 1 goto :fail

  echo [User %%U] done
  echo.
)

echo All users completed successfully.
exit /b 0

:fail
echo.
echo Failed with exit code %errorlevel%.
exit /b %errorlevel%
