# ==============================================================================
# Procfile — Render / Heroku / Railway process declaration
# ==============================================================================
# 'web' is the process type Render expects.
# setup.sh trains models on cold-start if outputs/ is empty,
# then hands off to Streamlit bound to the platform-assigned $PORT.
web: sh setup.sh && streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true --server.enableCORS=false
