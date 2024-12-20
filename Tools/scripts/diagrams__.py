from graphviz import Digraph

# Create a directed graph
graph = Digraph(format='png', graph_attr={'rankdir': 'LR'})  # Horizontal layout

# Define nodes with professional muted colors
graph.node('Input', 'INPUT', shape='rect', style='filled', color='#B0C4DE', fontname='Arial')  # Light Steel Blue
graph.node('Preprocess', 'PREPROCESS', shape='rect', style='filled', color='#D3D3D3', fontname='Arial')  # Light Gray

graph.node('ObjectDetection', 'OBJECT DETECTION', shape='rect', style='filled', color='#C1E1C1', fontname='Arial')  # Tea Green
graph.node('ObjectEmbedding', 'OBJECT EMBEDDING', shape='rect', style='filled', color='#A9A9A9', fontname='Arial')  # Dark Gray

graph.node('TextDetection', 'TEXT DETECTION', shape='rect', style='filled', color='#F0E68C', fontname='Arial')  # Khaki
graph.node('TextEmbedding', 'TEXT EMBEDDING', shape='rect', style='filled', color='#E6E6FA', fontname='Arial')  # Lavender

graph.node('ColourSpaceAnalysis', 'COLOUR SPACE ANALYSIS', shape='rect', style='filled', color='#FFDEAD', fontname='Arial')  # Navajo White
graph.node('ColourHistogramEmbedding', 'COLOUR HISTOGRAM EMBEDDING', shape='rect', style='filled', color='#D8BFD8', fontname='Arial')  # Thistle

graph.node('FullEmbedding', 'FULL EMBEDDING', shape='rect', style='filled', color='#B0E0E6', fontname='Arial')  # Powder Blue
graph.node('AttentionDense', 'ATTENTION AND DENSE LAYERS', shape='rect', style='filled', color='#F5F5F5', fontname='Arial')  # White Smoke

# Define edges
graph.edge('Input', 'Preprocess')

# Left branch
graph.edge('Preprocess', 'ObjectDetection')
graph.edge('ObjectDetection', 'ObjectEmbedding')

# Right branch
graph.edge('Preprocess', 'TextDetection')
graph.edge('TextDetection', 'TextEmbedding')

# Middle branch
graph.edge('Preprocess', 'ColourSpaceAnalysis')
graph.edge('ColourSpaceAnalysis', 'ColourHistogramEmbedding')

# Combine into Full Embedding
graph.edge('ObjectEmbedding', 'FullEmbedding')
graph.edge('TextEmbedding', 'FullEmbedding')
graph.edge('ColourHistogramEmbedding', 'FullEmbedding')

# From Full Embedding to final layers
graph.edge('FullEmbedding', 'AttentionDense')

# Render and visualize
graph.render('horizontal_graph_professional_colors', view=True)
