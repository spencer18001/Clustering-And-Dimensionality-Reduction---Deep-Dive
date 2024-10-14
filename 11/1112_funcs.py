import copy

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn.metrics import silhouette_score, adjusted_rand_score

import igraph as ig

ig.config['plotting.backend']='matplotlib'

def conductance(g_clust):
    
    """
    This function calculates clustering conductance as
    average of conductance of each cluster. Conductance for each cluster is 
    calculated as sum of all edges goind outside cluster divided by minimum
    between volume of cluster and volume of all other nodes.

    :param g_clust: Graph clustering object.
    :return: Clustering conductance.
    """    
            
    # extract graph object from clustering
    graph = g_clust.graph
    conductance_list = []
    all_node_ids = set(range(graph.vcount()))
    
    # calculate conductance for each cluster
    for clust_node_ids in g_clust:
        
        if len(clust_node_ids) > 1:
        
            other_node_ids = list(all_node_ids - set(clust_node_ids))
            
            external_edges = graph.es.select(_between=(clust_node_ids, other_node_ids))
            
            conductance_list.append(
                np.sum(external_edges['weight'])/np.min([
                    np.sum(graph.strength(clust_node_ids, weights='weight')),
                    np.sum(graph.strength(other_node_ids, weights='weight'))
                ])
            )
        
    return np.mean(conductance_list)


def average_internal_node_degree(g_clust):
    
    """
    Calculates internal node degree.

    :param g_clust: Graph clustering object.
    :return: Average internal node degree.
    """
    
    # get graph from igraph clusters
    internal_node_deg = []
    
    # iterate throug clusters
    for curr_subgraph in g_clust.subgraphs():
    
        node_degs = np.array(curr_subgraph.strength(weights='weight'))
        internal_node_deg.append(np.mean(node_degs))
            
    # return mean value for internal node degree
    return np.mean(internal_node_deg)


def print_clustering_stats(ig_clusters, min_cluster_size=0):
    
    """
    Prints number of nodes across clusters.

    :param ig_clusters: Graph clustering object.
    :param min_cluster_size: Adds minimum cluster size
    to prevent too many prints in case big number of 
    noise clusters is present.
    """

    # noise clusters are ones containing less fewer
    # than min_cluster_size
    num_noise_clusters = 0
    num_regular_clusters = 0
    
    # noise nodes are nodes belonging to noise
    # clusters
    num_noise_nodes = 0
        
    # if cluster is not noise then print cluster size
    # for noise clusters only print total number of 
    # clusters and total number of nodes
    for clust_id, cluster_nodes in enumerate(ig_clusters):
        if len(cluster_nodes)> min_cluster_size:
            print('Cluster {} size : {}'.format(clust_id, len(cluster_nodes)))
            num_regular_clusters += 1
        else:
            num_noise_clusters+=1
            num_noise_nodes += len(cluster_nodes)
            
    # print stats
    print('Num regular clusters {}'.format(num_regular_clusters))
    print('Num noise clusters {}'.format(num_noise_clusters))
    print('Num noise cluster nodes {}'.format(num_noise_nodes))
    
    
def display_network_clusters_labels(g_clust, layout, edge_width=None, vertex_size=None, min_size=0, color_edges=True, title=None, ax=None):
    
    """
    display_network_clusters_labels _summary_

    :param g_clust: Cluster graph object.
    :param layout: Graph layout.
    :param edge_width: Edge width, defaults to None.
    :param vertex_size: Vertex size, defaults to None.
    :param min_size: Clusters bellow this size will be considered noise clusters.
    :param color_edges: Whether to color edges in the graph, defaults to True.
    :param title: Graph title, defaults to None.
    :param ax: Matplotlib ax, defaults to None.
    
    :return: The graph object.
    """    
    
    
    plt.rcParams["figure.figsize"] = (10,10)
    
    # Initialize storage for regular clusters and noise clusters
    # noise clusters are ones noiseer than min_size
    noise_cluster = []
    regular_clusters = []
    
    noise_cluster_names = []
    regular_cluster_names = []
    
    # Make deep copy of the graph in 
    # order not to mess up the original
    # object
    g_clust = copy.deepcopy(g_clust)
    graph = g_clust.graph
    
    # Iterate through all the clusters and detect
    # noise clusters
    for clust_id, cluster_nodes in enumerate(g_clust):
                
        if len(cluster_nodes) > min_size:
            regular_clusters.append(cluster_nodes)
            regular_cluster_names.append(str(clust_id))
        else:
            noise_cluster += cluster_nodes
            noise_cluster_names.append(clust_id)
            
            
    # Get number of unique clusters
    num_clusters = len(regular_clusters)

    
    # Create collor pallete
    collor_palette = ig.ClusterColoringPalette(n=num_clusters)
        
    # Create legend
    custom_lines = []
    
    # Iterate through all communities
    
    # For each community, color internal edges and nodes
    # with same color
    for clust_id, cluster_nodes in enumerate(regular_clusters):
        
        # Cluster nodes is a list of all node ids
        # bellonging to the current community
        
        # Add node colors
        graph.vs[cluster_nodes]["color"] = [collor_palette[clust_id]]*len(cluster_nodes)
        
        # Add edge colors
        if color_edges:
            cluster_edges = graph.es.select(_within=cluster_nodes)
            cluster_edges['color'] = [collor_palette[clust_id]]*len(cluster_edges)
        
        # Create line object for legend
        custom_lines.append(
            Line2D([0], [0], color=collor_palette[clust_id], lw=4)
        )
        
    # Handle noise cluster
    if len(noise_cluster) > 0:
        
        clust_id += 1
        
        graph.vs[noise_cluster]["color"] = [(0,0,0,0.2)]*len(noise_cluster) 
        custom_lines.append(
            Line2D([0], [0], color=[0,0,0,1], lw=4)
        )
        legend_clust_names = regular_cluster_names + ['Outlier cluster']
        
    else:
    
        legend_clust_names = regular_cluster_names
        
    
    # Determine if matplotlib ax is provided
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 10))
    
    # draw network on the figure
    ig.plot(
        graph,
        edge_width=edge_width,
        vertex_size=vertex_size,
        layout=layout,
        target=ax
    )
    
    # add legend
    ax.legend(custom_lines, legend_clust_names)
    
    # add title
    if title:
        plt.title(title)


def plot_clust_stats(start_res, end_res, step, graph, original_data, original_labels, metric='euclidean'):
    
    """
    Performs graph clsutering for multiple resolutions.
    Plot multiple statistics for the clustering.

    :param start_res: Min resolution.
    :param end_res: Max resolution.
    :param step: Step.
    :param graph: Graph.
    :param original_data: Original data.
    :param original_labels: Ground truth labels.
    """
    
    # Set up resolution list and minimum cluster size
    resolution_list = np.arange(start_res, end_res, step)
    min_cluster_size=5
    stats_list = []

    # Cluster and record results and each resolution
    for resolution in resolution_list:
        
        print('Resolution {}'.format(resolution))
        ig_clusters = graph.community_multilevel(
            resolution=resolution,
            weights='weight'
        )
        
        # Deterct noise clusters by size
        noise_cluster = []
        regular_clusters = []
        
        for cluster_nodes in ig_clusters:
            
            if len(cluster_nodes) > min_cluster_size:
                regular_clusters.append(cluster_nodes)
            else:
                noise_cluster += cluster_nodes
            
        # Store results inside dict
        stats_list.append(
            {   
                'resolution':resolution,
                'modularity': graph.modularity(membership=ig_clusters.membership, weights='weight'),
                'conductance': conductance(ig_clusters),
                'avg_internal_degree': average_internal_node_degree(ig_clusters),
                'num_clusters': len(regular_clusters),
                'noise_clusters': len(ig_clusters) - len(regular_clusters),
                'noise_nodes': len(noise_cluster),
                'silhouette_score': silhouette_score(original_data, labels=ig_clusters.membership, metric=metric),
                'ARI': adjusted_rand_score(ig_clusters.membership, original_labels)
            }
        )

    # Transform list of dicts to dataframe
    stats_df = pd.DataFrame.from_dict(stats_list)

    # Plot the results

    # Split columns into resoluton and other columns
    res_col = 'resolution'
    stat_cols = stats_df.columns.tolist()
    stat_cols.remove(res_col)

    # Create plot gird
    ncols = 3
    n = len(stat_cols)
    nrows = int(np.ceil(n / ncols))

    fig, axs = plt.subplots(nrows, ncols, figsize=(12, 4*nrows))  # Adjust the size as needed

    # Perform plotting
    if axs.ndim > 1:
        axs = axs.flatten()

    for i, col in enumerate(stat_cols):
        axs[i].plot(stats_df[res_col], stats_df[col])
        axs[i].scatter(stats_df[res_col], stats_df[col])
        axs[i].set_title(col)
        axs[i].set_xlabel(res_col)
        axs[i].set_ylabel(col)

    plt.tight_layout()
    plt.show()
