from graphviz import Digraph

def draw_diagram():
    dot = Digraph()

    # Define nodes with more beautiful and sophisticated colors and square shapes
    dot.node('image', '<image>', style='filled', color='#1f77b4', fontcolor='white', shape='box')
    dot.node('yolo', '<Yolo Detection Discrete Model>', style='filled', color='#ff7f0e', fontcolor='white', shape='box')
    dot.node('bounding_boxes', 'Bounding Boxes', style='filled', color='#2ca02c', fontcolor='white', shape='box')
    dot.node('local_hsv', 'Local HSV', style='filled', color='#d62728', fontcolor='white', shape='box')
    dot.node('embedding_1024', '1024 Embedding', style='filled', color='#9467bd', fontcolor='white', shape='box')
    dot.node('tensor_shape', 'Tensor of shape #detections * 1024\nfor the localized HSV embedding', style='filled', color='#8c564b', fontcolor='white', shape='box')
    dot.node('easyocr', '<EasyOCR Discrete Model>', style='filled', color='#e377c2', fontcolor='white', shape='box')
    dot.node('bert_embedding', 'Bert Embedding', style='filled', color='#7f7f7f', fontcolor='white', shape='box')

    # Define edges
    dot.edge('image', 'yolo')
    dot.edge('yolo', 'bounding_boxes')
    dot.edge('bounding_boxes', 'local_hsv')
    dot.edge('local_hsv', 'embedding_1024')
    dot.edge('embedding_1024', 'tensor_shape')
    dot.edge('bounding_boxes', 'easyocr')
    dot.edge('easyocr', 'bert_embedding')

    # Set graph attributes for horizontal layout
    dot.attr(rankdir='LR')

    return dot

# Draw the diagram
diagram = draw_diagram()
diagram.render(filename='yolo_model_diagram_horizontal_colored_squares_beautiful', format='png', cleanup=True)
diagram.view()
