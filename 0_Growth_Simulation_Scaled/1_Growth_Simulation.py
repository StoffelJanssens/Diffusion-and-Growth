import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
import math


def arc_length(array_nodes):
    # Calculating the arc lengths at each node
    r_array = np.cumsum(np.sqrt(np.sum(np.power(np.roll(array_nodes, 1, axis=0) - array_nodes, 2), axis=1))[1:])
    # Adding the arc length at the first node
    r_array = np.insert(r_array, 0, 0.0)
    # Finding the arc length at the boundary
    r_omega = r_array[-1]
    return r_omega, r_array


def concentration_array(array_nodes, cap_c, concent_beta, diff_len_beta, cap_d):
    r_omega, r_array = arc_length(array_nodes)
    # Calculating the concentration at the boundary
    tan_h_r_omega = math.tanh(r_omega)
    d_beta_cap_d = diff_len_beta * cap_d
    concent_bound = (concent_beta + d_beta_cap_d * tan_h_r_omega) / (cap_c + d_beta_cap_d * tan_h_r_omega)
    # Calculating the concentration array
    array_concent = 1 + 2 * (concent_bound - 1) * np.exp(r_omega) * np.cosh(r_array) / (np.exp(2 * r_omega) + 1)
    # Returning the concentration array, which first has to be stacked for numpy purposes
    return np.stack((array_concent, array_concent), axis=1)


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


def topological_merging(array_nodes, min_dist, length_min_inclusion_array):
    # --- Avoiding self-intersection during growth ---
    # Calculating distances between all nodes. Check pdist function for more info.
    inter_dist_array = distance.pdist(array_nodes)

    # Transforming the 1D inter_dist_array to a NxN numpy matrix.
    distance_matrix = distance.squareform(inter_dist_array)

    # Finding all nodes that are in proximity and their index.
    proxi_nodes = np.where(distance_matrix < min_dist)

    # Preparation to remove neighbors out of proxi_nodes.
    # The absolute difference in index should be greater than one.
    diff_index_array = np.abs(proxi_nodes[0] - proxi_nodes[1])

    # Finding first index where the absolute difference in index is greater than one.
    res_search_gr_one = np.where(diff_index_array > 1)

    # The above routine is repeated till res_search_gr_one is empty.
    while np.size(res_search_gr_one) > 0:
        # All nodes between the pair of proxi nodes will be removed.
        node_1_index = proxi_nodes[0][res_search_gr_one[0][0]]
        node_2_index = proxi_nodes[1][res_search_gr_one[0][0]]
        x1, y1 = array_nodes[node_1_index]
        x2, y2 = array_nodes[node_2_index]

        # # Plotting a straight line between a pair of proxi nodes.
        # plt.plot([x1, x2], [y1, y2], color='pink')

        # Place node 1 at the point of intersection
        x1_moved = (x1 + x2) / 2
        y1_moved = (y1 + y2) / 2
        array_nodes[node_1_index] = [x1_moved, y1_moved]

        # Plotting of the portion that will be deleted if that portion is larger than length_min_inclusion_array.
        if node_2_index - node_1_index >= length_min_inclusion_array:
            hole_array = array_nodes[node_1_index + 1:node_2_index]
            plt.plot(hole_array[:, 0], hole_array[:, 1], color='black')

        # Deleting nodes
        array_nodes = np.delete(array_nodes, np.arange(node_1_index + 1, node_2_index, dtype=int), 0)

        # Repeat code above
        inter_dist_array = distance.pdist(array_nodes)
        distance_matrix = distance.squareform(inter_dist_array)
        proxi_nodes = np.where(distance_matrix < min_dist)
        diff_index_array = np.abs(proxi_nodes[0] - proxi_nodes[1])
        res_search_gr_one = np.where(diff_index_array > 1)

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


def simulation(array_nodes, arguments):
    green_lines_num, array_length_min_incl, cap_c, concent_beta, diff_len_beta, cap_d = arguments

    g = 0  # growth parameter
    plot_index = 1
    print('Progress = ', 0)
    growth_completed = False

    while not growth_completed:

        # Preventing negative y or z values.
        array_nodes = crossing_axis(array_nodes)

        # Remesh to prevent non-neighboring nodes to intersect.
        array_nodes = topological_merging(array_nodes, dist_min, array_length_min_incl)

        # Remesh by removing neighboring nodes that are too close.
        array_nodes = lower_bound_distance(array_nodes, dist_min)

        # Adding nodes.
        array_nodes = upper_bound_distance(array_nodes, dist_max)

        # Calculate the unit normal vectors of the nodes.
        node_norm_unit_array = unit_normals_nodes(array_nodes)

        # Concentration calculation.
        sigma = concentration_array(array_nodes, cap_c, concent_beta, diff_len_beta, cap_d)

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
            array_nodes = array_nodes + delta_g_c * sigma * node_norm_unit_array

        else:
            array_nodes = array_nodes + delta_g * sigma * node_norm_unit_array

        # Plotting.
        if g >= g_e * plot_index / green_lines_num:
            plt.plot(array_nodes[:, 0], array_nodes[:, 1], color='#7fcdbb')
            plot_index += 1

        print('Progress = ', g * 100 / g_e, '%')
    return array_nodes


# --- Input ---
# The following array represents the profile of $\partial\Omega^*_\alpha (g = 0)$
# Ensure that the last node lies on the scaled y-axis.
node_array = np.array([[0, 0], [4, 0], [3, 1], [3, 9], [4, 10], [8, 10], [9, 9], [9, 1], [8, 0],    # O
                       [11, 0], [11, 10], [13, 10], [13, 0],                                        # I
                       [15, 0], [15, 2], [19, 2], [19, 4], [16, 4], [15, 5], [15, 9], [16, 10],     # S
                       [21, 10], [21, 8], [17, 8], [17, 6], [20, 6], [21, 5], [21, 1], [20, 0],
                       [24, 0], [24, 8], [22, 8], [22, 10], [28, 10], [28, 8], [26, 8], [26, 0],    # T
                       [29, 0]], dtype=float)

# Set the number of profiles to be plotted at equally spaced values of g^*.
num_green_lines = 6

# Inclusions are plotted if they exceed the value of incl_array_length_min
incl_array_length_min = 20

# Parameters
c_cap = 1  # chemical potential-related C
sigma_beta = 0.5  # concentration-related \sigma^*_\beta
d_beta = 1  # boundary layer length related d^*_\beta
d_cap = 1  # diffusion related D

# Growth parameter value at the end of growth g_e equals unity means that one boundary layer is grown.
g_e = 4  # g^*_e

# Scaled version of | \bm{\eta}_\text{max} |
dist_max = 0.1

# --- End of Input ---

# --- Growth algorithm ---
# Minimum edge length.
dist_min = dist_max * 0.49

# Plotting the structure before growth.
plt.plot(node_array[:, 0], node_array[:, 1], color='#2c7fb8')

# Simulation
args = [num_green_lines,
        incl_array_length_min,
        c_cap,
        sigma_beta,
        d_beta,
        d_cap]
node_array = simulation(node_array, args)

# Last plots and saving data
np.save('node_array', node_array)
plt.xlabel('$y^*$')
plt.ylabel('$z^*$')
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.savefig('plot.pdf')
plt.show()
