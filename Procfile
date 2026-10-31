# ==============================================================================
# Procfile — Process declarations for Heroku / Railway / Render
# ==============================================================================
# Tells the platform which command to run when the app starts.
# The 'web' process type is auto-bound to the platform's $PORT variable.
web: sh setup.sh && streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true
