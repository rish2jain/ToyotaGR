#!/bin/bash

# RaceIQ Pro Dashboard Launcher

echo "🏁 Starting RaceIQ Pro Dashboard..."
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null
then
    echo "❌ Streamlit is not installed."
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Run the dashboard
streamlit run app.py
