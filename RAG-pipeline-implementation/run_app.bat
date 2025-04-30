@echo off
rem Setting environment variables to avoid PyTorch-Streamlit conflicts
set PYTHONPATH=%PYTHONPATH%;E:\Genai\genai-A3
set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
set OMP_NUM_THREADS=1
set STREAMLIT_SERVER_PORT=8501
set KMP_DUPLICATE_LIB_OK=TRUE

rem Running the Streamlit app with the fixed version
echo Starting Streamlit app with fixed version and environment settings...
streamlit run src/ui/app_fixed.py

pause