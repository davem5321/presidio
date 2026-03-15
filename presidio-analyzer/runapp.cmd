@echo off
cd /d "%~dp0"
poetry run python -m streamlit run streamlit_app.py --server.headless true %*
