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
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from bokeh.io import show, output_file
from numpy.random import random
from sklearn.utils import resample
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from bokeh.layouts import column, row
# from bokeh.charts import Bar

axis_map = {
    "gini": "gini",
    "entropy": "entropy"
}

desc = Div(text="""<h1>An Interactive Explorer for Churn Data</h1>
<p>
<a href="http://localhost:5006/index">Home Page</a> <br/>
</p>

<p>
The model allows user to work on random forest machine learning model
where the user can change model's hyperparameter and select features 
to drop. The user can also change the test size.
Based on the parameters user can see the changes in the graph in realtime
</p>
<p>
Prepared by <b>Rajat Kabra</b>.<br/>
Presented to <b>Prof. Chris Tseng</b>.<br/>
<b></b>.</p>
<br/>""")

df = pd.read_csv('C:\\Users\\rajat\\Desktop\\Master project\\churn.csv',names=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o'])

div = Div(text="""Your <a href="https://en.wikipedia.org/wiki/HTML">HTML</a>-supported text is initialized with the <b>text</b> argument.  The
remaining div arguments are <b>width</b> and <b>height</b>. For this example, those values
are <i>200</i> and <i>100</i> respectively.""",
width=200, height=100)

#columns = ['luxury_car_user','avg_dist','city_Astapor',"city_KingsLanding",'phone_Android','phone_iPhone']
df=df[1:]
y=df.o
x=df.drop('o',axis=1)

x_train_original,x_test_original,y_train_original,y_test_original=train_test_split(x,y,test_size=0.25)
#For standardizing data
clf = RandomForestClassifier(max_depth=10, random_state=1)
rfe = RFE(clf, 5)
rfe.fit(x_train_original,y_train_original)
predictions=rfe.predict(x_test_original)
print("Accuracy =", accuracy_score(y_test_original,predictions))
print(np.unique(predictions))
tn, fp, fn, tp = confusion_matrix(y_test_original,predictions).ravel()

fruits = ['True Positive', 'False Positive', 'False Negative', 'True Negative']
source = ColumnDataSource(data=dict(bins=fruits, counts=[1, 10, 20, 30]))
#fruits = [tp, fp, tn, fn]
# counts = [tp, fp, tn, fn]

source.data = dict(fruits=fruits, counts=[tp, fp, fn, tn])

p = figure(tools='crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,save,box_select,lasso_select',x_range=fruits, plot_height=450, title="Counts")
p.vbar(x='fruits', top='counts', width=0.9, source=source, legend="fruits",
       line_color='white',fill_color=factor_cmap('fruits', palette=Spectral6, factors=fruits))

labels = LabelSet(x='fruits', y='counts', text='counts', level='glyph',
        x_offset=-15, y_offset=0, source=source, render_mode='canvas')
p.add_layout(labels)      
       
p.title.text = "Model Accuracy %f" % accuracy_score(y_test_original,predictions)
tsize = Slider(title="Test Size", start=0.05, end=0.50, value=0.2, step=0.05)
input = Slider(title="Depth", start=1, end=50, value=3, step=1)
ent = Select(title="Entropy", options=sorted(axis_map.keys()), value="gini")
button = Button(label="Submit Parameters", button_type="success")
button2 = Button(label="Play_Depth", button_type="success")
multi_select = MultiSelect(title="Features to drop:", value=[],options=[('a','avg_dist'),('b','avg_rating_by_driver'),('c','avg_rating_of_driver'),('d','avg_surge'),('e','surge_pct'),('f','trips_in_first_30_days'),('g','luxury_car_user'),('h','weekday_pct'),('i','city_Astapor'),('k','city_Winterfell'),('l','phone_Android'),('m','phone_iPhone'),('n','phone_no_phone')])
p.add_tools(HoverTool(tooltips = [
    ("Count", "@counts"),
]))
def update_points():
    E = ent.value
    N = int(input.value)
    T = float(tsize.value)
    ms= multi_select.value
    print(ms)
    ms = [str(i) for i in ms]
    print(ms)
    df = pd.read_csv('C:\\Users\\rajat\\Desktop\\Master project\\churn.csv',names=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o'])
    df.head()
    df=df[1:]
    y=df.o
    x=df.drop('o',axis=1)
    for dr in ms:
        x=x.drop(dr,axis=1)
    x_train_original,x_test_original,y_train_original,y_test_original=train_test_split(x,y,test_size=T)
    clf = RandomForestClassifier(criterion=E,max_depth=N, random_state=1)
    clf.fit(x_train_original,y_train_original)
    predictions=clf.predict(x_test_original)
    print("Accuracy =", accuracy_score(y_test_original,predictions))
    print(np.unique(predictions))
    tn, fp, fn, tp = confusion_matrix(y_test_original,predictions).ravel()
    print("True Negative:",tn)
    print("False Positive:",fp)
    print("False Negative:",fn)
    print("True Positive:",tp)
    source.data=dict(fruits=fruits, counts=[tp, fp, fn, tn])
    p.yaxis.bounds = (0,100000)
    p.title.text = "Model Accuracy %f" % accuracy_score(y_test_original,predictions)
button.on_click(update_points)


# sizing_mode = 'fixed'  # 'scale_width' also looks nice with this example

# def update_pointsss(attrname, old, new):
#     E = ent.value
#     N = int(input.value)
#     T = float(tsize.value)
#     ms= multi_select.value
#     print(ms)
#     ms = [str(i) for i in ms]
#     print(ms)
#     df = pd.read_csv('C:\\Users\\rajat\\Desktop\\Master Project\\churn.csv',names=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o'])
#     df.head()
#     df=df[1:]
#     y=df.o
#     x=df.drop('o',axis=1)
#     for dr in ms:
#         x=x.drop(dr,axis=1)
#     x_train_original,x_test_original,y_train_original,y_test_original=train_test_split(x,y,test_size=T)
#     clf = RandomForestClassifier(criterion=E,max_depth=N, random_state=1)
#     clf.fit(x_train_original,y_train_original)
#     predictions=clf.predict(x_test_original)
#     print("Accuracy =", accuracy_score(y_test_original,predictions))
#     print(np.unique(predictions))
#     tn, fp, fn, tp = confusion_matrix(y_test_original,predictions).ravel()
#     print("True Negative:",tn)
#     print("False Positive:",fp)
#     print("False Negative:",fn)
#     print("True Positive:",tp)
#     source.data=dict(fruits=fruits, counts=[tp, fp, fn, tn])
#     p.yaxis.bounds = (0,100000)
#     p.title.text = "Model Accuracy %f" % accuracy_score(y_test_original,predictions)
# input.on_change('value',update_pointsss)



def animate_update():
    ne=input.value+1
    input.value=ne
    update_points()
    if ne > 50:
        input.value = 1
    
def animate_update1():
    ne1=tsize.value+0.05
    tsize.value=ne1
    update_points()
    if ne1 > 0.5:
        tsize.value = 0.05


def animate():
    if button1.label == 'Play_Depth':
        button1.label = 'Pause'
        curdoc().add_periodic_callback(animate_update, 2000)
    else:
        button1.label = 'Play_Depth'
        curdoc().remove_periodic_callback(animate_update)

def animate2():
    if button2.label == 'Play_Test':
        button2.label = 'Pause_Test'
        curdoc().add_periodic_callback(animate_update1, 3000)
    else:
        button2.label = 'Play_Test'
        curdoc().remove_periodic_callback(animate_update1)

button1 = Button(label='Play_Depth', width=60)
button1.on_click(animate)
button2= Button(label='Play_Test', width=60)
button2.on_click(animate2)

layout = layout([
    [desc],
    [ent, input,tsize,multi_select],
    [p,button,button1,button2]
])
output_file("panning.html")

#update()  # initial load of the data
 
curdoc().add_root(layout)
curdoc().title = "Churn"
print("1")