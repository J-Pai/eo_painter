# eo_painter
Tool used to paint clusters on a dot-based heatmap images.

## Setup
Make sure to install python3 prior to running the following steps.

```
# First time setup.
py [or python3] -m venv venv

# WINDOWS
source venv/Scripts/activate
venv/Scripts/pip install -r requirements.windows.txt

# LINUX
sudo apt install scrot python3-tk python3-dev
source venv/bin/activate
venv/bin/pip install -r requirements.linux.txt

# Execution
venv/Scripts/python cluster_painter.py
```
