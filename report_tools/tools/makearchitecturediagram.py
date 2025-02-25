from graphviz import Digraph

def create_full_image_model_with_weighted_class_names():
    dot = Digraph(comment='FullImageModel with Weighted Class Names')

    # Input nodes
    dot.node('A', 'text_embeddings\n(batch_size, 12, 128)', shape='box', style='filled', color='lightblue')
    dot.node('B', 'color_histograms\n(batch_size, 128, 3)', shape='box', style='filled', color='lightgreen')
    dot.node('C', 'class_names_vectors\n(batch_size, 128, 1)', shape='box', style='filled', color='lightcoral')
    dot.node('D', 'confidences\n(batch_size, 128, 1)', shape='box', style='filled', color='lightyellow')

    # Dense and embedding layers for inputs
    dot.node('E', 'text_embedding_dense\n(batch_size, 128)', shape='box', style='filled', color='lightblue')
    dot.node('F', 'color_histogram_dense\n(batch_size, 128)', shape='box', style='filled', color='lightgreen')
    dot.node('G', 'confidence_dense\n(batch_size, 128)', shape='box', style='filled', color='lightyellow')
    
    # Class name embedding and one-hot encoding
    dot.node('H', 'class_name_embedding\n(batch_size, 128, 1000)', shape='box', style='filled', color='lightcoral')
    dot.node('I', 'One-hot Encoding\n(batch_size, 128, 1000)', shape='box', style='dotted')

    # Weighted class name embeddings (using confidences)
    dot.node('J', 'Weighted Class Name Embeddings\n(batch_size, 128, 1000)', shape='box', style='filled', color='lightpink')

    # Multi-Head Attention layer
    dot.node('K', 'Multi-Head Attention\n(batch_size, 128, attention_dim)', shape='box', style='filled', color='lightgrey')

    # Concatenate features after attention
    dot.node('L', 'Concatenate Features\n(batch_size, 128, X)', shape='box', style='filled', color='lightpink')

    # Dense layers
    dot.node('M', 'concat_dense1\n(batch_size, 256)', shape='box', style='filled', color='lightblue')
    dot.node('N', 'concat_dense2\n(batch_size, 128)', shape='box', style='filled', color='lightblue')

    # Output layer
    dot.node('O', 'Output Layer\n(batch_size, 2)', shape='box', style='filled', color='lightgreen')

    # Add edges between layers to represent the flow of data
    dot.edge('A', 'E')
    dot.edge('B', 'F')
    dot.edge('C', 'H')
    dot.edge('H', 'I')
    dot.edge('D', 'G')
    dot.edge('E', 'K')
    dot.edge('I', 'J')
    dot.edge('J', 'K')  # Weighted class name embeddings flow into attention
    dot.edge('K', 'L')  # Attention output flows to concatenation with color histograms and confidences
    dot.edge('F', 'L')  # Color histograms flow to concatenation after attention
    dot.edge('G', 'L')  # Confidences flow to concatenation after attention
    dot.edge('L', 'M')
    dot.edge('M', 'N')
    dot.edge('N', 'O')

    # Render the diagram to a file (e.g., PNG)
    dot.render('full_image_model_with_weighted_class_names', format='png', cleanup=True)
    print("Corrected diagram with weighted class names generated successfully!")

# Generate and save the corrected diagram
create_full_image_model_with_weighted_class_names()