from typing import Union, Dict, Any, Optional, Type, List

from torch import nn
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GAT

from modules.hmpn import AbstractMessagePassingBase
from modules.hmpn.common.hmpn_util import unpack_homogeneous_features
from modules.hmpn.homogeneous.homogeneous_graph_assertions import HomogeneousGraphAssertions

"""
c.f. https://pytorch-geometric.readthedocs.io/en/2.0.1/_modules/torch_geometric/nn/conv/gat_conv.html#GATConv,
https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.GAT.html?highlight=gat

GAT Equations vs. Implementation
===============================

Equations
---------
1. The attention coefficients are derived using the concatenation of transformed node features:

.. math::

\alpha_{i,j} \propto \exp\left(\mathbf{a}^{\top} \left[\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j\right]\right)

Where:

.. math::

\mathbf{a} = \left[\mathbf{a}_1 \, \Vert \, \mathbf{a}_2\right]

Expanding, we get:

.. math::

\alpha_{i,j} \propto \exp\left(\mathbf{a}_1^{\top} \mathbf{\Theta}\mathbf{x}_i + \mathbf{a}_2^{\top} \mathbf{\Theta}\mathbf{x}_j\right)


Implementation
--------------
1. The code computes attention coefficients by summing the individual attention values of source and target nodes:

.. math::

\alpha_{i,j} \propto \exp(\alpha_i + \alpha_j)

Where:

.. math::

\alpha_i = \mathbf{\Theta}\mathbf{x}_i \cdot \mathbf{a}_{\text{src}}

.. math::

\alpha_j = \mathbf{\Theta}\mathbf{x}_j \cdot \mathbf{a}_{\text{dst}}

For the two forms to be equivalent:

.. math::

\mathbf{a}_{\text{src}} = \mathbf{a}_1

.. math::

\mathbf{a}_{\text{dst}} = \mathbf{a}_2


Key Takeaways
-------------
- The implementation captures the essence of the original attention mechanism using separate weight vectors for source and target nodes.
- This decomposition can lead to computational efficiency improvements.
- The equivalence is based on the assumption that the concatenated weight vector in the original formulation can be split into separate components for source and target nodes, potentially reducing the parameter space.
"""


def get_gat_from_graph(*, example_graph: Data,
                       latent_dimension: int,
                       base_config: Dict[str, Any],
                       device: Optional = None):
    """
    Args:
        example_graph:
        latent_dimension:
        base_config:
        device:

    Returns:

    """
    in_node_features = example_graph.x.shape[1]
    in_edge_features = example_graph.edge_attr.shape[1]
    stack_config = base_config.get("stack")
    return GAT(in_channels=in_node_features,
               edge_dim=in_edge_features,
               hidden_channels=latent_dimension,
               num_layers=stack_config["num_steps"],
               heads=1,
               v2=False,
               ).to(device)


class Noop():
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return args[0]


class GATWrapper(AbstractMessagePassingBase):

    def __init__(self, *, in_node_features: int,
                 in_edge_features: int,
                 latent_dimension: int,
                 scatter_reduce_strs: Union[List[str], str],
                 stack_config: Dict[str, Any],
                 embedding_config: Dict[str, Any],
                 unpack_output: bool,
                 edge_dropout: float = 0.0,
                 flip_edges_for_nodes: bool = False,
                 create_graph_copy: bool = True,
                 assert_graph_shapes: bool = True,
                 node_name: str = "node"):
        """
        Args:
            in_node_features:
                Node feature input size for a homogeneous graph.
                Node features may have size 0, in which case an empty input graph of the appropriate shape/batch_size
                is expected and the initial embeddings are learned constants
            in_edge_features:
                Edge feature input size for a homogeneous graph.
                Edge features may have size 0, in which case an empty input graph of the appropriate shape/batch_size
                is expected and the initial embeddings are learned constants
            latent_dimension:
                Latent dimension of the network. All modules internally operate with latent vectors of this dimension
            scatter_reduce_strs:
                Names of the scatter reduce to use to aggregate messages of the same type.
                Can be multiple of "sum", "mean", "max", "min", "std"
                e.g. ["sum","max"]
            stack_config:
                Configuration of the stack of GNN steps. See the documentation of the stack for more information.
            embedding_config:
                Configuration of the embedding stack (can be empty by choosing None, resulting in linear embeddings).
            flip_edges_for_nodes:
                If true, the edge features are flipped for each node, i.e. the edge features of the incoming edges
                are concatenated to the node features.
                If false, the edge features are not flipped, i.e. the edge features of the outgoing edges are
                concatenated to the node features.
            edge_dropout:
                Dropout probability for the edges. Removes edges from the graph with the given probability.
            unpack_output:
                If true, the output of the forward pass is unpacked into a tuple of (node_features, edge_features).
                If false, the output of the forward pass is the raw graph.
            create_graph_copy:
                If True, a copy of the input graph is created and modified in-place.
                If False, the input graph is modified in-place.
            assert_graph_shapes:
                If True, the input graph is checked for consistency with the input shapes.
                If False, the input graph is not checked for consistency with the input shapes.
            node_name:
                Name of the node. Used to convert the output of the forward pass to a dictionary
        """
        super().__init__(in_node_features=in_node_features,
                         in_edge_features=in_edge_features,
                         latent_dimension=latent_dimension,
                         embedding_config=embedding_config,
                         scatter_reduce_strs=scatter_reduce_strs,
                         edge_dropout=edge_dropout,
                         unpack_output=unpack_output,
                         create_graph_copy=create_graph_copy,
                         assert_graph_shapes=assert_graph_shapes
                         )
        assert isinstance(stack_config, dict), f"Expected stack_config to be a dict, but got {type(stack_config)}"
        assert embedding_config is None, f"Expected embedding_config to be None, but got {type(embedding_config)}"
        assert flip_edges_for_nodes is False, f"Expected flip_edges_for_nodes to be False got {type(flip_edges_for_nodes)}"

        if isinstance(scatter_reduce_strs, list):
            assert len(scatter_reduce_strs) == 1, (f"Expected scatter_reduce_strs to be a list of length 1, "
                                                   f"got {scatter_reduce_strs}")
            scatter_reduce_strs = scatter_reduce_strs[0]
        assert scatter_reduce_strs in ["sum",
                                       "mean"], (f"Expected scatter_reduce_strs to be 'sum' or 'mean', "
                                                 f"got {scatter_reduce_strs}")
        self._node_name = node_name

        regularization_config = stack_config.get("mlp").get("regularization")
        dropout = regularization_config.get("dropout")

        # create message passing stack
        self.gat = GAT(in_channels=latent_dimension,
                       edge_dim=latent_dimension,
                       hidden_channels=latent_dimension,
                       num_layers=stack_config["num_steps"],
                       heads=2,
                       v2=True,
                       act=stack_config.get("mlp").get("activation_function"),
                       dropout=dropout,
                       )

    def _get_assertions(self) -> Type[HomogeneousGraphAssertions]:
        return HomogeneousGraphAssertions

    @staticmethod
    def _get_input_embedding_type() -> Type:
        return Noop

    def unpack_features(self, graph: Batch) -> Batch:
        return unpack_homogeneous_features(graph, node_name=self._node_name)

    def _process_encoded_graph(self, graph: Batch) -> Batch:
        graph.x = self.gat(graph.x, graph.edge_index, edge_attr=graph.edge_attr)
