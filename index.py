from os.path import dirname, join
from bokeh.models.widgets import Slider, Select, TextInput, MultiSelect, Dropdown, Button
import numpy as np
import pandas as pd
# import pandas.io.sql as psql
from bokeh.plotting import figure
from bokeh.layouts import layout, widgetbox
from bokeh.models import ColumnDataSource, HoverTool, Div, LabelSet
from bokeh.io import curdoc
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score
from bokeh.transform import factor_cmap
from bokeh.palettes import Spectral6
from bokeh.io import show, output_file
from bokeh.layouts import column, row
# from bokeh.charts import Bar


desc = Div(text="""<!DOCTYPE html>
<html>
<head>
<style>
a:link {
    color: green;
    background-color: transparent;
    text-decoration: none;
}
a:visited {
    color: pink;
    background-color: transparent;
    text-decoration: none;
}
a:hover {
    color: red;
    background-color: transparent;
    text-decoration: underline;
}
a:active {
    color: yellow;
    background-color: transparent;
    text-decoration: underline;
}
</style>
</head>
<body>

<p><a href="http://localhost:5006/code_RandomForest"><font size="5">Random Forest</font></a> <br/></p>
<p><a href="http://localhost:5006/code_SVM"><font size="5">SVM </font></a> <br/></p>
<p><a href="http://localhost:5006/code_MLP"><font size="5">MLP</font></a> <br/></p>
<p><a href="http://localhost:5006/code_Keras"><font size="5">Keras</font></a> <br/></p>

</body>
</html>
""")
layout = layout([
    [desc]
])
output_file("panning.html")

#update()  # initial load of the data
 
curdoc().add_root(layout)
curdoc().title = "Home"