import graphviz

# Create a new Graphviz Digraph with better styling
dot = graphviz.Digraph(format='png', engine='dot')

# Global graph styling for a professional look
dot.attr(dpi='300', fontname='Helvetica', fontsize='12', style='solid', rankdir='LR')

# Add the Street Level Image node with an actual image (no label over the image)
dot.node('A', '', shape='rect', width='2', height='2', 
         image='SampleLocation.png', style='filled', fillcolor='#f0f0f0')

# Add the Predictive Model Psuedo Ensemble node with box styling
with dot.subgraph() as s:
    s.attr(rank='same', style='filled', fillcolor='#e0e0e0')  # Define subgraph's color
    s.node('B', 'Predictive Model Psuedo Ensemble', shape='rect', width='3', height='2', 
           style='filled', fillcolor='#e0e0e0', fontname='Helvetica')

    # Add Model A and Model B inside the "Predictive Model Psuedo Ensemble" box
    s.node('B1', 'Model A', shape='rect', width='2', height='1', style='filled', fillcolor='#e0f7fa')
    s.node('B2', 'Model B', shape='rect', width='2', height='1', style='filled', fillcolor='#f0f0f0')

    # Optionally, you can arrange Model A and B inside in a row or column format by using 'rank'
    s.attr(rankdir='TB')  # Top to Bottom layout for sub-nodes

# Add the Result Latitude Longitude node with box styling
dot.node('C', 'Result Latitude Longitude', shape='rect', width='2', height='1.2', 
         style='filled', fillcolor='#e0f7fa', fontname='Helvetica')

# Add edges (connections) with labels
dot.edge('A', 'B', label='input', color='black', fontname='Helvetica', fontsize='10')
dot.edge('B', 'C', label='output', color='black', fontname='Helvetica', fontsize='10')

# Render the graph to a file (e.g., a PNG image)
dot.render('predictive_model_diagram_with_nested_models')

# Display the diagram
dot.view('predictive_model_diagram_with_nested_models')