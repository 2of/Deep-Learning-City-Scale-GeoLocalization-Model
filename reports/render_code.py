from graphviz import Digraph

def create_pipeline_diagram():
    dot = Digraph("Code Structure", format="png")

    # Set transparent background
    dot.attr(bgcolor="transparent")

    # Ingest Stage
    dot.node("Ingest", "Ingest\n(Images + Locations)", shape="box", style="filled", fillcolor="#FFDDC1")

    # Processing Steps
    dot.node("YOLO", "YOLO Object Detection", shape="box", style="filled", fillcolor="#FFABAB")
    dot.node("TextDetect", "Text Detection", shape="box", style="filled", fillcolor="#FFC3A0")
    dot.node("HSV", "Color Analysis (HSV)", shape="box", style="filled", fillcolor="#D5AAFF")
    
    # YOLO branch for signs
    dot.node("YOLO-Signs", "Signs Detected", shape="box", style="filled", fillcolor="#85E3FF")
    dot.node("TextSigns", "Sign Text Detection", shape="box", style="filled", fillcolor="#B9FBC0")
    dot.node("HSVSigns", "Sign Color Analysis (HSV)", shape="box", style="filled", fillcolor="#FFCFD2")

    # Datasets
    dot.node("MainDataset", "Main Dataset", shape="box", style="filled", fillcolor="#A0C4FF")
    dot.node("SignsDataset", "Signs Dataset", shape="box", style="filled", fillcolor="#BDB2FF")

    # Edges - Main Processing Flow
    dot.edge("Ingest", "YOLO")
    dot.edge("Ingest", "TextDetect")
    dot.edge("Ingest", "HSV")
    
    # YOLO to Sign Processing
    dot.edge("YOLO", "YOLO-Signs", label="Signs")
    dot.edge("YOLO-Signs", "TextSigns")
    dot.edge("YOLO-Signs", "HSVSigns")

    # Final Dataset Storage
    dot.edge("YOLO", "MainDataset")
    dot.edge("TextDetect", "MainDataset")
    dot.edge("HSV", "MainDataset")
    dot.edge("TextSigns", "SignsDataset")
    dot.edge("HSVSigns", "SignsDataset")

    return dot

# Generate and render the diagram
diagram = create_pipeline_diagram()
diagram.render("code_structure_diagram", view=True)


from graphviz import Digraph

def create_prediction_diagram():
    dot = Digraph("Prediction Pass", format="png")

    # Set transparent background
    dot.attr(bgcolor="transparent")

    # Nodes
    dot.node("Unseen", "Unseen Image", shape="box", style="filled", fillcolor="#FFDDC1")
    dot.node("Ingest", "Ingest", shape="box", style="filled", fillcolor="#FFABAB")

    # Embeddings
    dot.node("SignEmbedding", "Sign Embedding", shape="box", style="filled", fillcolor="#85E3FF")
    dot.node("OverallEmbedding", "Overall Embedding", shape="box", style="filled", fillcolor="#A0C4FF")

    # Networks
    dot.node("SignGeoNet", "Sign GeoLocalization Network", shape="box", style="filled", fillcolor="#B9FBC0")
    dot.node("FullGeoNet", "Full Image GeoLocalization Network", shape="box", style="filled", fillcolor="#D5AAFF")

    # Final Output
    dot.node("FinalOutput", "Final Average Output", shape="box", style="filled", fillcolor="#FFCFD2")

    # Edges (Flow of Processing)
    dot.edge("Unseen", "Ingest")
    dot.edge("Ingest", "SignEmbedding")
    dot.edge("Ingest", "OverallEmbedding")
    dot.edge("SignEmbedding", "SignGeoNet")
    dot.edge("OverallEmbedding", "FullGeoNet")
    dot.edge("SignGeoNet", "FinalOutput")
    dot.edge("FullGeoNet", "FinalOutput")

    return dot

# Generate and render the diagram
diagram = create_prediction_diagram()
diagram.render("prediction_pass_diagram", view=True)