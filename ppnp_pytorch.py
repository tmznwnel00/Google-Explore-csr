import logging

from ppnp.pytorch import PPNP
from ppnp.pytorch.training import train_model
from ppnp.pytorch.earlystopping import stopping_args
from ppnp.pytorch.propagation import PPRExact, PPRPowerIteration
# from ppnp.data.io import load_dataset
from ppnp.data.io import networkx_to_sparsegraph
from graph import G

logging.basicConfig(
        format='%(asctime)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO)

graph_name = "snu_ds"
graph = networkx_to_sparsegraph(G, label_name="label")
print("finished sparsegraph")
graph.standardize(select_lcc=True)
print(graph)

prop_ppnp = PPRExact(graph.adj_matrix, alpha=0.1)
prop_appnp = PPRPowerIteration(graph.adj_matrix, alpha=0.1, niter=10)

model_args = {
    'hiddenunits': [64],
    'drop_prob': 0.5,
    'propagation': prop_appnp}

idx_split_args = {'ntrain_per_class': 20, 'nstopping': 500, 'nknown': 1500, 'seed': 2413340114}
reg_lambda = 5e-3
learning_rate = 0.01

test = True
device = 'cpu'
print_interval = 20

model, result = train_model(
        graph_name, PPNP, graph, model_args, learning_rate, reg_lambda,
        idx_split_args, stopping_args, test, device, None, print_interval)