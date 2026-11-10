@echo off
REM Inicia Label Studio para corregir las cajas. Abrir luego http://localhost:8080
REM Usuario: admin@rosas.local   Clave: definida en LS_PASSWORD
set LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
set LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=C:\Users\lab22\rosas_dataset\work
set LABEL_STUDIO_ENABLE_LEGACY_API_TOKEN=true
"C:\Users\lab22\rosas_dataset\.venv_ls\Scripts\label-studio.exe" start --no-browser --port 8080 ^
  --data-dir "C:\Users\lab22\rosas_dataset\labelstudio" ^
  --username admin@rosas.local --password %LS_PASSWORD% --user-token %LS_TOKEN%
