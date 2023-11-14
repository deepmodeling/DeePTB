from dptb.nn.deeptb import DPTB
from dptb.nn.sktb import SKTB

def build_model(model_options):
    """
    this method provide a unified interfaces to use the graph nn module classes defined in dptb/nn, 
    to construct a graph neural network model for different usages. For examples:
     - build a model for based on descriptors need:
        1. a descriptor model
        2. a embedding model
        3. a residual or FNN model
        4. a quantity related model, such as a aggregation model for energy, grad model for forces, 
            SKrotation for SK hamiltonian, and E3rotation for E3 hamiltonian.
     - build a model for based on Graph Neural Network is simular, since we restrict all models take AtomicData dict
        as input and output, we only need to replace the descriptor model and embedding model with a Graph Neural Network model.
    """

    # process the model_options
    
    model = None


    return model

