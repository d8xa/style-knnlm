import pandas as pd

def pick_layer(dataset, type):
    """Helper function to pick a dataset layer from the nested datasets."""

    current = dataset
    while True:
        if isinstance(current, type):
            return current
        if not hasattr(current, "dataset"): 
            break
        current = current.dataset

    return None

def list_layers(dataset):
    """Helper function to list all layers of nested datasets."""

    layers = []
    current = dataset
    while True:
        layers += [current]
        if not hasattr(current, "dataset"): 
            break
        current = current.dataset
    
    return layers

def summarize_layers(dataset):
    """Returns info of layers in a nested datset."""

    layers = reversed(list_layers(dataset))
    return pd.DataFrame(
        [(type(layer).__name__, len(layer.sizes), sum(layer.sizes), i) 
        for i,layer in enumerate(layers)],
        columns=["name", "samples", "tokens", "depth"]
    ).set_index("depth").sort_index(ascending=False)
