import numpy as np
from scipy.spatial import distance


def arc_length(array_nodes):
    # Calculating the arc lengths at each node
    r_array = np.cumsum(np.sqrt(np.sum(np.power(np.roll(array_nodes, 1, axis=0) - array_nodes, 2), axis=1))[1:])
    # Adding the arc length at the first node
    r_array = np.insert(r_array, 0, 0.0)
    # Finding the arc length at the boundary
    r_omega = r_array[-1]
    return r_omega, r_array


def remeshing(array_nodes, num_nodes):
    """
    Strategy
    --------
    A new node (c) should be placed as follows: ](a), (b)]
    Node (a) is located at a arc length value that is smaller than that of node (b).
    Node (b) is located next to (a).
    """
    r_omega, r_array = arc_length(array_nodes)

    # Calculating the arc lengths at which the (c) nodes should be placed
    r_c_array = np.linspace(0, r_omega, num_nodes)

    # finding (b) nodes related to new nodes
    node_index_array_b = np.zeros(num_nodes, dtype=int)
    for ind, arc_len in enumerate(r_c_array):
        node_index_array_b[ind] = np.argmax(r_array > arc_len)

    # correction for last node
    node_index_array_b[-1] = np.shape(array_nodes)[0] - 1

    # arc length of (a) and (b)
    r_a_array = np.zeros(num_nodes, dtype=float)
    r_b_array = np.zeros(num_nodes, dtype=float)

    for ind, node_location in enumerate(node_index_array_b):
        r_a_array[ind] = r_array[node_location - 1]
        r_b_array[ind] = r_array[node_location]

    # distance between (a) and new node (c)
    distance_a_c = r_c_array - r_a_array

    # distance between (a) and (b)
    distance_a_b = r_b_array - r_a_array

    # ratio distances
    ratio_distances = distance_a_c / distance_a_b
    ratio_distances = np.stack((ratio_distances, ratio_distances), axis=1)

    # values (a) nodes
    array_nodes_a = np.zeros((num_nodes, 2), dtype=float)
    array_nodes_b = np.zeros((num_nodes, 2), dtype=float)
    for ind, location in enumerate(node_index_array_b):
        array_nodes_a[ind, 0], array_nodes_a[ind, 1] = array_nodes[location - 1]
        array_nodes_b[ind, 0], array_nodes_b[ind, 1] = array_nodes[location]

    # values (c) nodes
    array_nodes_c = array_nodes_a + (array_nodes_b - array_nodes_a) * ratio_distances

    return array_nodes_c


def concentration_array_r_bound_large(array_nodes, sigma_bound):
    r_omega, r_array = arc_length(array_nodes)

    # Calculating the concentration array
    array_sigma = 1 + 2 * (sigma_bound - 1) * np.exp(r_omega) * np.cosh(r_array) / (np.exp(2 * r_omega) + 1)

    # Returning the concentration array, which first has to be stacked for numpy purposes
    return np.stack((array_sigma, array_sigma), axis=1)


def dist_edge_sq(array_nodes, ind_i, ind_j):
    return (array_nodes[ind_i][0] - array_nodes[ind_j][0]) ** 2 + (array_nodes[ind_i][1] - array_nodes[ind_j][1]) ** 2


def unit_normals_nodes(array_nodes):
    # The outward surface unit normals at nodes are calculated in this function
    edge_array = np.roll(array_nodes, -1, axis=0) - array_nodes  # 2D array
    edge_array_mag_prep = np.sqrt(np.sum(np.power(edge_array, 2), axis=1))  # 1D array
    edge_array_mag = np.stack((edge_array_mag_prep, edge_array_mag_prep), axis=1)  # 2D array
    edge_array_unit = edge_array / edge_array_mag  # unit tangent
    edge_norm_unit_array = np.stack((-edge_array_unit[:, 1], edge_array_unit[:, 0]), axis=1)  # edge unit normal array

    node_norm_array = edge_norm_unit_array + np.roll(edge_norm_unit_array, 1, axis=0)
    node_norm_array_mag_prep = np.sqrt(np.sum(np.power(node_norm_array, 2), axis=1))  # 1D array
    node_norm_array_mag = np.stack((node_norm_array_mag_prep, node_norm_array_mag_prep), axis=1)  # 2D array

    # preventing division by zero
    node_norm_array_mag[0] = [1, 1]
    node_norm_array_mag[-1] = [1, 1]

    array_node_norm_unit = node_norm_array / node_norm_array_mag

    array_node_norm_unit[0] = [.0, 1.0]
    array_node_norm_unit[-1] = [1.0, .0]

    return array_node_norm_unit


def crossing_axis(array_nodes):
    # Preventing negative coordinates for nodes
    # -----------------------------------------
    # --- Preventing negative y-values ---
    # Find last node with negative y-val
    # Calculate where the node zero should be located
    # Remove all preceding nodes including that last node
    # Set the y-val of the first node to zero
    array_neg_x = np.argwhere(array_nodes[:, 0] < 0)
    if np.size(array_neg_x) > 0:
        loc_last_neg_x = array_neg_x[-1][0]
        x1, y1 = array_nodes[loc_last_neg_x]
        x2, y2 = array_nodes[loc_last_neg_x + 1]
        y0 = (y1 - y2) / (x2 - x1) * x1 + y1

        # removing nodes
        array_nodes = array_nodes[loc_last_neg_x + 1:]

        # placing first node
        array_nodes[0] = [0, y0]

    # --- Preventing negative z-values ---
    # Find first node with negative z-val
    # Remove all subsequent nodes including that first node
    # Set the z-val of the last node to zero
    array_neg_y = np.argwhere(array_nodes[:, 1] < 0)
    if np.size(array_neg_y) > 0:
        loc_first_neg_y = array_neg_y[0][0]
        x1, y1 = array_nodes[loc_first_neg_y]
        x2, y2 = array_nodes[loc_first_neg_y - 1]
        x0 = (x1 - x2) / (y2 - y1) * y1 + x1

        # removing nodes
        array_nodes = array_nodes[:loc_first_neg_y]
        array_nodes[-1] = [x0, 0]

    return array_nodes


def calculate_proximity_nodes(array_nodes, min_dist):
    # Calculating distances between all nodes and transforming into NxN matrix
    inter_dist_array = distance.pdist(array_nodes)
    distance_matrix = distance.squareform(inter_dist_array)

    # Finding all nodes that are in proximity and their index
    proxi_nodes = np.where(distance_matrix < min_dist)

    # Finding the absolute difference in index and locating first valid pair
    diff_index_array = np.abs(proxi_nodes[0] - proxi_nodes[1])
    res_search_gr_one = np.where(diff_index_array > 1)

    return proxi_nodes, res_search_gr_one


def topological_merging(array_nodes, min_dist):
    # Initial calculation of proximity nodes
    proxi_nodes, res_search_gr_one = calculate_proximity_nodes(array_nodes, min_dist)

    # Loop until no valid proximity node pair is found
    while np.size(res_search_gr_one) > 0:
        # All nodes between the pair of proximity nodes will be removed
        node_1_index = proxi_nodes[0][res_search_gr_one[0][0]]
        node_2_index = proxi_nodes[1][res_search_gr_one[0][0]]
        x1, y1 = array_nodes[node_1_index]
        x2, y2 = array_nodes[node_2_index]

        # Place node 1 at the midpoint between node 1 and node 2
        x1_moved = (x1 + x2) / 2
        y1_moved = (y1 + y2) / 2
        array_nodes[node_1_index] = [x1_moved, y1_moved]

        # Deleting nodes between node 1 and node 2
        array_nodes = np.delete(array_nodes, np.arange(node_1_index + 1, node_2_index, dtype=int), 0)

        # Recalculate proximity nodes after deletion
        proxi_nodes, res_search_gr_one = calculate_proximity_nodes(array_nodes, min_dist)

    return array_nodes


def lower_bound_distance(array_nodes, min_dist):
    # Saving the last node for later
    node_xn_1 = array_nodes[-1][0]

    # Calculating the square of min_dist
    min_dist_sq = min_dist ** 2

    # Obtaining the size of array_nodes
    size_array = np.shape(array_nodes)[0]

    # Algorithm to remove nodes from 0 to n-1
    i = 0
    j = 1
    node_list = []
    while i < size_array - 1:
        while dist_edge_sq(array_nodes, i, j) < min_dist_sq:
            node_list.append(j)
            if j == size_array - 1:
                break
            j += 1
        i = j
        j += 1

    # Moving nodes for symmetry
    for node_index in node_list:
        # Not moving node 0!
        if node_index > 1:
            x1, y1 = array_nodes[node_index - 1]
            x2, y2 = array_nodes[node_index]
            x1_moved = (x1 + x2) / 2
            y1_moved = (y1 + y2) / 2
            array_nodes[node_index - 1] = [x1_moved, y1_moved]

    # Delete all the nodes found by the algorithm
    array_nodes = np.delete(array_nodes, node_list, 0)

    # If the last node is removed, the z-value of the new last node is larger than zero.
    # If the last node is removed, it is placed back.
    if array_nodes[-1][1] > 0:
        array_nodes = np.append(array_nodes, [[node_xn_1, 0]], axis=0)

    return array_nodes


def upper_bound_distance(array_nodes, max_dist):
    # Calculating the array for edge lengths
    array_edge_l = np.sqrt(np.sum(np.power(np.roll(array_nodes, 1, axis=0) - array_nodes, 2), axis=1))[1:]

    # Finding the locations where the maximum internode distance is surpassed
    edge_loc_array = np.ravel(np.argwhere(array_edge_l > max_dist))
    size_edge_loc_array = np.size(edge_loc_array)

    # Adding all the nodes in the new node array
    while size_edge_loc_array > 0:
        # Calculating the positions of new nodes
        positions_new_nodes = (array_nodes[edge_loc_array] + array_nodes[edge_loc_array + 1]) / 2
        # Inserting the new nodes
        array_nodes = np.insert(array_nodes, edge_loc_array + 1, positions_new_nodes, axis=0)

        # Repeat code above.
        array_edge_l = np.sqrt(np.sum(np.power(np.roll(array_nodes, 1, axis=0) - array_nodes, 2), axis=1))[1:]
        edge_loc_array = np.ravel(np.argwhere(array_edge_l > max_dist))
        size_edge_loc_array = np.size(edge_loc_array)

    return array_nodes


def simulation(params, args):
    dif_length, sigma_omega_scaled = params

    # calling the args and scale with diffusion length
    node_array, g_e, dist_max = args

    node_array = node_array / dif_length
    g_e = g_e / dif_length
    dist_max = dist_max / dif_length

    # Minimum distance between nodes.
    dist_min = dist_max * 0.49

    g = 0  # growth parameter $g_\alpha$, not growth $g$.
    growth_completed = False
    while not growth_completed:

        # Preventing negative y or z values.
        node_array = crossing_axis(node_array)

        # Remesh to prevent non-neighboring nodes to intersect.
        node_array = topological_merging(node_array, dist_min)

        # Remesh by removing neighboring nodes that are too close.
        node_array = lower_bound_distance(node_array, dist_min)

        # Adding nodes.
        node_array = upper_bound_distance(node_array, dist_max)

        # Calculate the unit normal vectors of the nodes.
        node_norm_unit_array = unit_normals_nodes(node_array)

        # Concentration calculation.
        sigma = concentration_array_r_bound_large(node_array, sigma_omega_scaled)

        # Growth.
        sigma_first = sigma[0][0]
        sigma_last = sigma[-1][0]
        if sigma_first <= sigma_last:
            delta_g = dist_min / sigma_last
        else:
            delta_g = dist_min / sigma_first

        g += delta_g

        # correcting for shooting over g_e
        if g >= g_e:
            growth_completed = True
            delta_g_c = delta_g - (g - g_e)
            g = g_e
            node_array = node_array + delta_g_c * sigma * node_norm_unit_array

        else:
            node_array = node_array + delta_g * sigma * node_norm_unit_array

        print('Progress = ', g * 100 / g_e, '%')

    return node_array * dif_length


# --- Input ---
n_g_0_given = 1600  # n(g = 0) of the given profile
g_e = 1.0  # g_e
d_a = 1.0  # d_\alpha
sigma_o_sigma_a = 2.0  # \sigma_omega/sigma_\alpha
r_o_g_0 = 5.0  # r_\omega(g = 0)
# --- End of Input ---

# generate profile before growth (g = 0)
node_array_before = np.array([[0, 0], [r_o_g_0, 0]], dtype=float)
node_array_before = remeshing(node_array_before, n_g_0_given)
np.save('given_prof_g_0.npy', node_array_before)

# generate profile after growth (g = g_e)
param_list_given = [d_a, sigma_o_sigma_a]
arg_list_given = [node_array_before, g_e, r_o_g_0 / n_g_0_given]
node_array_after = simulation(param_list_given, arg_list_given)
np.save('given_prof_g_e.npy', node_array_after)
