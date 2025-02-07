import holoviews as hv
import graphviz
from holoviews import opts

# Initialize the holoviews extension
hv.extension('bokeh')

# Create a Graphviz Digraph object
dot = graphviz.Digraph(format='png', engine='neato')

# Nodes: Image and the various models
dot.node('Image', 'Image', shape='ellipse', style='filled', fillcolor='lightblue', width='1.5', height='1')
dot.node('YOLOv11', 'YOLOv11 Base Model', shape='ellipse', style='filled', fillcolor='lightgreen', width='1.5', height='1')
dot.node('Crosswalk', 'Crosswalk Model', shape='ellipse', style='filled', fillcolor='lightyellow', width='1.5', height='1')
dot.node('Hydrant', 'Hydrant Model', shape='ellipse', style='filled', fillcolor='lightcoral', width='1.5', height='1')
dot.node('Signage', 'Signage Model', shape='ellipse', style='filled', fillcolor='lightpink', width='1.5', height='1')
dot.node('StreetLamp', 'StreetLamp Model', shape='ellipse', style='filled', fillcolor='lightgoldenrodyellow', width='1.5', height='1')

# Edges: Arrows coming out from "Image" to each model
dot.edge('Image', 'YOLOv11', label="YOLOv11 Detection")
dot.edge('Image', 'Crosswalk', label="Crosswalk Detection")
dot.edge('Image', 'Hydrant', label="Hydrant Detection")
dot.edge('Image', 'Signage', label="Signage Detection")
dot.edge('Image', 'StreetLamp', label="StreetLamp Detection")

# Render the graph to a PNG file
dot.render('image_to_models_flow', cleanup=True)

# Display the generated image using Holoviews
img = hv.Image('image_to_models_flow.png')
img.opts(width=800, height=400)
