from graphviz import Digraph

def draw_model_diagram():
    dot = Digraph()

    # Define the nodes with shapes and colors
    dot.node('text_embeddings', 'Text Embeddings\n[..., 12, 128]', shape='box', style='filled', color='lightblue')
    dot.node('color_histograms', 'Color Histograms\n[..., 3, 256]', shape='box', style='filled', color='lightgreen')
    dot.node('reshape', 'Reshape\n[..., 2, 128]', shape='box', style='filled', color='lightyellow')
    dot.node('concat', 'Concatenate\n[..., 14, 128]', shape='box', style='filled', color='lightcoral')
    dot.node('conv1', 'Conv2D (64 filters)', shape='box', style='filled', color='lightpink')
    dot.node('conv2', 'Conv2D (128 filters)', shape='box', style='filled', color='lightcyan')
    dot.node('flatten', 'Flatten', shape='box', style='filled', color='lightgrey')
    dot.node('dense1', 'Dense (256 units)', shape='box', style='filled', color='lightsalmon')
    dot.node('dense2', 'Dense (128 units)', shape='box', style='filled', color='lightgoldenrodyellow')
    dot.node('output_layer', 'Output Layer (2 units)', shape='box', style='filled', color='lightsteelblue')

    # Define the edges
    dot.edge('text_embeddings', 'reshape')
    dot.edge('color_histograms', 'reshape')
    dot.edge('reshape', 'concat')
    dot.edge('concat', 'conv1')
    dot.edge('conv1', 'conv2')
    dot.edge('conv2', 'flatten')
    dot.edge('flatten', 'dense1')
    dot.edge('dense1', 'dense2')
    dot.edge('dense2', 'output_layer')

    # Set the graph attributes for horizontal layout and square shape
    dot.attr(rankdir='LR')  # Left to Right layout
    dot.attr(size='10,10')  # Square shape

    # Render the diagram
    dot.render('model_diagram_colored_horizontal_square_batch_size', format='png', view=True)

draw_model_diagram()