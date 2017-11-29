from os.path import dirname, join
from bokeh.models.widgets import Slider, Select, TextInput, MultiSelect, Dropdown, Button
import numpy as np
import pandas as pd
# import pandas.io.sql as psql
from bokeh.layouts import layout, widgetbox
from bokeh.plotting import figure
from sklearn.svm import SVC
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
    "linear": "linear",
    "poly": "poly",
    "rbf": "rbf",
    "sigmoid": "sigmoid"
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
clf = MLPClassifier(hidden_layer_sizes=(30,30,9),max_iter=1000)
# rfe = RFE(clf, 5)
clf.fit(x_train_original,y_train_original)
predictions=clf.predict(x_test_original)
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
layer = Slider(title="Layer 1", start=10, end=50, value=10, step=10)
layer1 = Slider(title="Layer 2", start=10, end=50, value=10, step=10)
layer2 = Slider(title="Layer 3", start=10, end=50, value=10, step=10)
layer3 = Slider(title="Layer 4", start=10, end=50, value=10, step=10)
maxiter = Slider(title="Iterations", start=100, end=3000, value=100, step=100)
button = Button(label="Submit Parameters", button_type="success")
button2 = Button(label="Play_Maximum_Iterations", button_type="success")
multi_select = MultiSelect(title="Features to drop:", value=[],options=[('a','avg_dist'),('b','avg_rating_by_driver'),('c','avg_rating_of_driver'),('d','avg_surge'),('e','surge_pct'),('f','trips_in_first_30_days'),('g','luxury_car_user'),('h','weekday_pct'),('i','city_Astapor'),('k','city_Winterfell'),('l','phone_Android'),('m','phone_iPhone'),('n','phone_no_phone')])
p.add_tools(HoverTool(tooltips = [
    ("Count", "@counts"),
]))
def update_points():
    # E = ent.value
    l1 = int(layer.value)
    l2 = int(layer1.value)
    l3 = int(layer2.value)
    l4 = int(layer3.value)
    T = float(tsize.value)
    iter=int(maxiter.value)
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
    clf = MLPClassifier(hidden_layer_sizes=(l1,l2,l3,l4),max_iter=iter)
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
layer = Slider(title="Layer 1", start=10, end=50, value=10, step=10)
layer1 = Slider(title="Layer 2", start=10, end=50, value=10, step=10)
layer2 = Slider(title="Layer 3", start=10, end=50, value=10, step=10)
layer3 = Slider(title="Layer 4", start=10, end=50, value=10, step=10)
maxiter = Slider(title="Iterations", start=100, end=3000, value=100, step=100)

def animate_update1():
    ne1=tsize.value+0.05
    tsize.value=ne1
    update_points()
    if ne1 > 0.5:
        tsize.value = 0.05


def animatelayer1():
    if button1.label == 'Play_Layer1':
        button1.label = 'Pause'
        curdoc().add_periodic_callback(animate_update, 10000)
    else:
        button1.label = 'Play_Layer1'
        curdoc().remove_periodic_callback(animate_update)
def animate_update():
    ne=layer.value+5
    layer.value=ne
    update_points()
    if ne > 50:
        input.value = 10

def animatelayer2():
    if button1.label == 'Play_Layer2':
        button1.label = 'Pause'
        curdoc().add_periodic_callback(animate_update2, 10000)
    else:
        button1.label = 'Play_Layer2'
        curdoc().remove_periodic_callback(animate_update2)
def animate_update2():
    ne=layer1.value+5
    layer1.value=ne
    update_points()
    if ne > 50:
        layer1.value = 10
def animatelayer3():
    if button1.label == 'Play_Layer3':
        button1.label = 'Pause'
        curdoc().add_periodic_callback(animate_update3, 10000)
    else:
        button1.label = 'Play_Layer3'
        curdoc().remove_periodic_callback(animate_update3)
def animate_update3():
    ne=layer2.value+5
    layer2.value=ne
    update_points()
    if ne > 50:
        layer2.value = 10
def animatelayer4():
    if button1.label == 'Play_Layer4':
        button1.label = 'Pause'
        curdoc().add_periodic_callback(animate_update4, 10000)
    else:
        button1.label = 'Play_Layer4'
        curdoc().remove_periodic_callback(animate_update4)
def animate_update4():
    ne=layer3.value+5
    layer3.value=ne
    update_points()
    if ne > 50:
        input.value = 10
def animate2():
    if button2.label == 'Play_Test':
        button2.label = 'Pause_Test'
        curdoc().add_periodic_callback(animate_update1, 3000)
    else:
        button2.label = 'Play_Test'
        curdoc().remove_periodic_callback(animate_update1)
def animate_Iter():
    ne=maxiter.value+100
    maxiter.value=ne
    update_points()
    if ne > 2500:
        input.value = 100
def animateIter():
    if buttonIter.label == 'Play_Iteration':
        buttonIter.label = 'Pause_Iteration'
        curdoc().add_periodic_callback(animate_Iter, 3000)
    else:
        buttonIter.label = 'Play_Iteration'
        curdoc().remove_periodic_callback(animate_Iter)
button1 = Button(label='Play_Layer1')
button1.on_click(animatelayer1)
button3 = Button(label='Play_Layer2')
button3.on_click(animatelayer2)
button4 = Button(label='Play_Layer3')
button4.on_click(animatelayer3)
button5 = Button(label='Play_Layer4')
button5.on_click(animatelayer4)
buttonIter = Button(label='Play_Iteration')
buttonIter.on_click(animateIter)
button2= Button(label='Play_Test')
button2.on_click(animate2)
controls = [tsize,layer,layer1,layer2,layer3,maxiter,button1, button3,button4,button5,button2]
sizing_mode = 'fixed'  # 'scale_width' also looks nice with this example
inputs = widgetbox(*controls, sizing_mode=sizing_mode)
# layout = layout([
#     [desc,p],
#     [tsize,layer,layer1, button1, ],
#     [layer3,maxiter,button3,button4],
#     [button5,multi_select,button2],
#     [button]
# ])
layout = layout([
    [desc,multi_select,button],
    [inputs, p],
], sizing_mode=sizing_mode)

output_file("panning.html")

#update()  # initial load of the data
 
curdoc().add_root(layout)
curdoc().title = "Churn"