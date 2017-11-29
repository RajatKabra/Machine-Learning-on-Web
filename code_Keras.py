import sys
print(sys.version)
from os.path import dirname, join
from bokeh.models.widgets import Slider, Select, TextInput, MultiSelect, Dropdown, Button
import numpy as np
from sklearn.preprocessing import StandardScaler
# import pandas.io.sql as psql
from bokeh.plotting import figure
from bokeh.layouts import layout, widgetbox
from bokeh.models import ColumnDataSource, HoverTool, Div, LabelSet
from bokeh.io import curdoc
from bokeh.transform import factor_cmap
from bokeh.palettes import Spectral6
from bokeh.io import show, output_file
from sklearn.preprocessing import StandardScaler
from bokeh.layouts import column, row
# from bokeh.charts import Barfrom keras.models import Sequential
np.random.seed(7)
import pandas as pd
from keras.models import Sequential
# Import `Dense` from `keras.layers`
from keras.layers import Dense
from sklearn.model_selection import train_test_split
axis_map = {
    "rel": "relu",
    "sigmoid": "sigmoid"
}
desc = Div(text="""<h1>An Interactive Explorer for Wine Data</h1>
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
div = Div(text="""Your <a href="https://en.wikipedia.org/wiki/HTML">HTML</a>-supported text is initialized with the <b>text</b> argument.  The
remaining div arguments are <b>width</b> and <b>height</b>. For this example, those values
are <i>200</i> and <i>100</i> respectively.""",
width=200, height=100)

# fix random seed for reproducibility

# Read in white wine data 
white = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep=';')
# Read in red wine data 
red = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=';')
red['type'] = 1
# Add `type` column to `white` with value 0
white['type'] = 0
# Append `white` to `red`
wines = red.append(white, ignore_index=True)

import numpy as np
# Specify the data 
X=wines.iloc[:,0:11]
# Specify the target labels and flatten the array 
y=np.ravel(wines.type)
# Split the data up in train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Define the scaler 
scaler = StandardScaler().fit(X_train)
# Scale the train set
X_train = scaler.transform(X_train)
# Scale the test set
X_test = scaler.transform(X_test)
# Import `Sequential` from `keras.models`
# Initialize the constructor
model = Sequential()
# Add an input layer 
model.add(Dense(12, activation='relu', input_shape=(11,)))
# Add one hidden layer 
model.add(Dense(8, activation='relu'))
# Add an output layer 
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=100, verbose=1)
y_pred = model.predict(X_test)
y_pred=np.rint(y_pred)
y_pred=y_pred.astype(int)

score = model.evaluate(X_test, y_test,verbose=1)
print(score)
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score
# Confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()


#print("Accuracy =", accuracy_score(y_test_original,predictions))
#print(np.unique(predictions))

fruits = ['True Red Wine', 'False Red Wine', 'False White Wine', 'True White Wine']
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
       
#p.title.text = "Model Accuracy %f" % accuracy_score(y_test,y_pred)
tsize = Slider(title="Test Size", start=0.05, end=0.50, value=0.2, step=0.05)
layer1 = Slider(title="Layer 1", start=2, end=10, value=8, step=1)
active1 = Select(title="Activation 1", options=sorted(axis_map.keys()), value="relu")
layer2 = Slider(title="Layer 2", start=2, end=10, value=8, step=1)
active2 = Select(title="Activation 2", options=sorted(axis_map.keys()), value="relu")
layer3 = Slider(title="Layer 3", start=2, end=10, value=8, step=1)
active3 = Select(title="Activation 3", options=sorted(axis_map.keys()), value="relu")
epochslide = Slider(title="Epoch", start=50, end=2500, value=50, step=50)
batchsize = Slider(title="Batch size", start=100, end=1500, value=100, step=100)
button = Button(label="Submit Parameters", button_type="success")
button2 = Button(label="Play_Depth", button_type="success")
multi_select = MultiSelect(title="Features to drop:", value=[],options=[('a','avg_dist'),('b','avg_rating_by_driver'),('c','avg_rating_of_driver'),('d','avg_surge'),('e','surge_pct'),('f','trips_in_first_30_days'),('g','luxury_car_user'),('h','weekday_pct'),('i','city_Astapor'),('k','city_Winterfell'),('l','phone_Android'),('m','phone_iPhone'),('n','phone_no_phone')])
p.add_tools(HoverTool(tooltips = [
    ("Count", "@counts"),
]))
def update_points():
	T = float(tsize.value)
	l1=int(layer1.value)
	l2=int(layer2.value)
	l3=int(layer3.value)
	a1=active1.value
	a2=active2.value
	a3=active3.value
	ep=int(epochslide.value)
	b=int(batchsize.value)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=T, random_state=42)
	# Define the scaler 
	scaler = StandardScaler().fit(X_train)
	# Scale the train set
	X_train = scaler.transform(X_train)
	# Scale the test set
	X_test = scaler.transform(X_test)
	# Import `Sequential` from `keras.models`
	# Initialize the constructor
	model = Sequential()
	# Add an input layer 
	model.add(Dense(l1, activation=a1, input_shape=(11,)))
	# Add one hidden layer 
	model.add(Dense(l2, activation=a2))
	model.add(Dense(l3, activation=a3))
	# Add an output layer 
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy',
	              optimizer='adam',
	              metrics=['accuracy'])
	model.fit(X_train, y_train, epochs=ep, batch_size=b, verbose=1)
	y_pred = model.predict(X_test)
	y_pred=np.rint(y_pred)
	y_pred=y_pred.astype(int)

	score = model.evaluate(X_test, y_test,verbose=1)
	print(score)
	# Confusion matrix
	tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()
	source.data = dict(fruits=fruits, counts=[tp, fp, fn, tn])
	p.yaxis.bounds = (0,100000)
	# p.title.text = "Model Accuracy %f" % score
button.on_click(update_points)



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

controls = [tsize,layer1,active1,layer2,active2,layer3,active3, epochslide, batchsize]
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
    [desc,button],
    [inputs, p],
], sizing_mode=sizing_mode)

output_file("panning.html")

#update()  # initial load of the data
 
curdoc().add_root(layout)
curdoc().title = "Churn"