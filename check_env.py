import sys
import shutil
import platform

import numpy
import pandas
import plotly
import matplotlib
import streamlit

print("Python      :", sys.version.split()[0], platform.machine())
print("numpy       :", numpy.__version__)
print("pandas      :", pandas.__version__)
print("plotly      :", plotly.__version__)
print("matplotlib  :", matplotlib.__version__)
print("streamlit   :", streamlit.__version__)
print("ffmpeg      :", shutil.which("ffmpeg") or "NOT FOUND")
print("ffplay      :", shutil.which("ffplay") or "NOT FOUND")
print("ENVIRONMENT OK" if shutil.which("ffmpeg") and shutil.which("ffplay") else "FIX FFMPEG PATH")