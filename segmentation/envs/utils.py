import numpy as np
import numpy.matlib
from scipy.spatial import distance
import math
import open3d as o3d
import matplotlib.pyplot as plt
from pyntcloud import PyntCloud
import pandas as pd

def get_base_colors():
    return np.array([
        [0,0,1],
        [0,1,0],
        [1,0,0],
        [1,1,0],
        [1,0,1],
        [0,1,1]])

def generate_colors(max_colors):
    colors = get_base_colors()
    return _generate_colors(max_colors, colors)

def generate_heat_colors(max_colors):
    colors = np.array([[1,0,0]])
    return _generate_colors(max_colors, colors)
        
def _generate_colors(max_colors, colors):
    tmp_colors = np.array(colors, copy=True)
    n_colors = colors.shape[0]
    
    iter = int(np.ceil(max_colors / n_colors))
    for i in range(1,iter):
        tmp_colors = np.vstack((tmp_colors, (i/iter) * colors))
    colors = tmp_colors
    return colors
        
def preprocess_state(x_t, env):
    x_t[:,:3] = x_t[:,:3] / np.absolute(env.max_room)
    x_t[:,env.label_col] = x_t[:,env.label_col] / env.orig_labels.shape[0]
    return x_t

def np_vec_to_mat(vec):
	return np.reshape(vec, (vec.shape[0], 1))
    
def np_vec_to_mat_T(vec):
	return np.reshape(vec, (1, vec.shape[0]))

def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j][1] < arr[j+1][1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr
    
def get_angles(A, B):
    inner = np.inner(A, B)
    dots = np.diagonal(inner)
    angles = np.arccos(np.clip(dots, -1, 1))
    return angles

def NN(points):
    D = distance.squareform(distance.pdist(points))
    closest = np.argsort(D, axis=1)
    return closest

def NN_o3d(pc, pcd, k_neighbours = 30):
    flann = o3d.geometry.KDTreeFlann(pcd)
    nns = np.zeros((pc.shape[0], k_neighbours))
    for i in range(pc.shape[0]):
        query = pc[i,:]
        _, indexes, _ = flann.search_knn_vector_3d(query=query, knn=k_neighbours)
        indexes = np.asarray(indexes)
        nns[i] = indexes
    return nns, flann

def NN_point_o3d(flann, query, k_neighbours = 30):
    k, indexes, neighbours = flann.search_knn_vector_3d(query=query, knn=k_neighbours)
    return np.asarray(indexes), np.asarray(neighbours)

def flip_normals(all_points, normals, label_col, angle_threshold = math.pi/2):
    wall_idxs = np.where(all_points[:,label_col] == 0)[0]
    wall_normals = normals[wall_idxs,:]
    wall_points = all_points[wall_idxs,:label_col]
    wall_vecs = wall_points / np.linalg.norm(wall_points, axis=1)[:,None]
    angles = get_angles(wall_vecs, wall_normals)
    flip_idxs = np.where(angles < angle_threshold)[0]
    wall_normals[flip_idxs,:] *= -1
    normals[wall_idxs,:] = wall_normals
    return normals
    
def compute_error(all_points, labels, label_col, orig_labels, orig_indices, orig_counts):
    assignments = {}
    
    n_errornous_points = 0
    intervals = []

    label_values, label_counts = np.unique(labels, return_counts = True)
    orig_values, orig_label_counts = np.unique(all_points[:,label_col] , return_counts = True)

    diff = np.absolute(orig_label_counts.shape[0] - label_counts.shape[0])

    # walls: orig_labels[:6], objects: orig_label[6:]
    for i in range(orig_labels.shape[0]):
        idx = orig_indices[i] # start idx of a original label
        count = orig_counts[i] # how many original labels follow after the start idx
        obj_labels = labels[idx : idx + count] # np array of estimated in the range of a original label
        
        # which estimated labels exist in the range of a original label and how often they occur
        sorted_labels, obj_counts = np.unique(obj_labels, return_counts=True) 
        # ( int, int, np.array, np.array )
        intervals.append( (idx, count, sorted_labels, obj_counts, i) )

    #intervals = bubble_sort(intervals)
    n_unlabelled_points = 0
    u_wall = 0
    u_obj = 0
    e_wall = 0
    e_obj = 0
    for i in range(len(intervals)):
        obj_idx = intervals[i][0] # orig
        len_points = intervals[i][1] # orig
        sorted_labels = intervals[i][2] # estimated
        counts = intervals[i][3] # estimated
        orig_label = intervals[i][4] # orig
        #print(orig_label)

        # estimated labels over the range of a true segment
        obj_labels = labels[obj_idx:obj_idx+len_points]

        u_points = len(np.argwhere(obj_labels == -1))
        n_unlabelled_points += u_points
        if orig_label < 6: # wall
            u_wall += u_points
        else: # object
            u_obj += u_points

        # check if a label is already assigned
        is_already_assigned = True
        # check if a label is available
        no_label_available = False

        while is_already_assigned:
            # no more label available for assignment - this happens if there are less cluster than predicted
            if counts.size == 0:
                no_label_available = True
                break
            # get the index of the most frequent cluster
            j = np.argmax(counts)
            # get the most frequent cluster label
            if sorted_labels[j] == -1:
                is_already_assigned = True
            else:
                chosen_label = sorted_labels[j]
                # check if label is already assigned
                is_already_assigned = chosen_label in assignments.values()
            if(is_already_assigned):
                # if so delete the label and take the second most label and so forth
                sorted_labels = np.delete(sorted_labels, j)
                counts = np.delete(counts, j)
            else:
                break
        # if there are no more labels, consider all the points as misclustered
        # n_labels > n_orig_labels
        e_points=0
        if no_label_available:
            false_points = np.argwhere(obj_labels != -1)
            e_points=len(false_points)
            n_errornous_points += e_points
        else:
            # save the assignet label for next iterations
            assignments[i] = chosen_label
            # filter the objec	ts with the wrong labels
            false_points = np.argwhere(obj_labels != chosen_label)
            false_points = np.argwhere(obj_labels[false_points] != -1)
            # increment the n_errornous_points for every false point
            e_points=len(false_points)
            n_errornous_points += e_points
        if orig_label < 6: # wall
            e_wall += e_points 
        else: # object
            e_obj += e_points
    return n_errornous_points, n_unlabelled_points, diff, u_wall, u_obj, e_wall, e_obj

def compute_reward(n_points, n_errornous_points, n_unlabelled_points, false_labels_punishment=1.0, unlabelled_punishment=1.0):
    reward = 1
    reward -= ((n_errornous_points/ n_points) * false_labels_punishment)
    reward -= ((n_unlabelled_points / n_points) * unlabelled_punishment)
    reward = np.maximum(reward, 0)
    return reward
    
def compute_weighted_reward(n_points, 
        e_wall, u_wall, n_wall, 
        e_obj, u_obj, n_objs, 
        wall_weight=0.5, obj_weight=0.5):
    wall_error = ((u_wall + e_wall)/n_wall)
    wall_reward = 1 - wall_error
    obj_error = ((u_obj + e_obj)/n_objs)
    obj_reward = 1 - obj_error
    #print("wall_error: ", wall_error)
    #print("obj_error: ", obj_error)
    #print("u_obj: ", u_obj/n_objs)
    #print("e_obj: ", e_obj/n_objs)
    
    reward = wall_weight * wall_reward + obj_weight * obj_reward
    reward = np.maximum(reward, 0)
    return reward
    
def segment(all_points, flann, query_p, labels, label_col, nns, K, angle_threshold, curvature_threshold, region):    
    K = int(K)
    indexes, _ = NN_point_o3d(flann = flann, query = query_p, k_neighbours = K)
    
    candidate_idx = int(indexes[0])
    
    no_candidate=True
    for i in range(1, indexes.shape[0]):
        if labels[candidate_idx] == -1:
            no_candidate = False
            break
        candidate_idx = indexes[i]
    
    if no_candidate:
        return labels, no_candidate
    
    seed_point_idx = candidate_idx

    labels[seed_point_idx] = region
    seed_point_idxs = [seed_point_idx]
    while(len(seed_point_idxs) != 0):
        seed_point_idx = seed_point_idxs[0]
        seed_normal = all_points[seed_point_idx, 3:6]
        
        # get neighbours of seed point 
        nn_idxs = nns[seed_point_idx, 1:(K+1)]
        
        neighbour_labels = labels[nn_idxs]
        # identify points that are not assigned to a label
        free_n_idxs = np.where(neighbour_labels == -1)[0]
        # if all neighbours are already assigned to a region
        if free_n_idxs.shape[0] == 0:
            if len(seed_point_idxs) > 0:
                # try next seed point
                del seed_point_idxs[0]
            continue

        # extract the unlabelled idxs
        nn_idxs = nn_idxs[free_n_idxs]
        # extract the unlabelled rows
        row_neighbours = all_points[nn_idxs, :]
        # normals of the unlabelled points
        n_normals = row_neighbours[:, 3:6]
        
        seed_normals = np.zeros((n_normals.shape[0], 3)) + seed_normal
        
        # compute the angles between the seed_normal and the normals of the neighbours
        angles = get_angles(seed_normals, n_normals)
        
        region_idxs = nn_idxs[angles < angle_threshold]
        if region_idxs.size == 0:
            if len(seed_point_idxs) > 0:
                del seed_point_idxs[0]
            continue
        labels[region_idxs] = region
        if len(seed_point_idxs) > 0:
            del seed_point_idxs[0]
        
        region_idxs = region_idxs[all_points[region_idxs, 6] < curvature_threshold]
        
        if region_idxs.shape[0] > 0:
            seed_point_idxs.extend(region_idxs.tolist())
    return labels, no_candidate

def render_point_cloud(all_points, pcd, labels, colors, query_ps):
    values = np.unique(labels)
    n_labels = values.shape[0]
    col_mat = np.zeros((all_points.shape[0], 3))
    for i in range(n_labels):
        label = values[i]
        color = colors[i,:]
        idx = np.where(labels == label)[0]
        col_mat[idx, :] = color
    pcd.colors = o3d.utility.Vector3dVector(col_mat)

    if len(query_ps) > 0:
        pcds = []
        pcds.append(pcd)
        for i in range(len(query_ps)):
            query_p = query_ps[i]
            query_p = np.transpose(np_vec_to_mat(query_p))
            q_pcd = o3d.geometry.PointCloud()
            q_pcd.points = o3d.utility.Vector3dVector(query_p)
            q_pcd.colors = o3d.utility.Vector3dVector(np.array([[0.5, 0.5, 0.5]]))
            pcds.append(q_pcd)
        pcds.append(coordinate_system())
        o3d.visualization.draw_geometries(pcds)
    else:
        o3d.visualization.draw_geometries([pcd, coordinate_system()])
        
def is_nan_or_inf(x):
    return np.any(np.isnan(x)) or np.any(np.isinf(x))
    
def get_indices(len_P, sample_size, len_wall_idxs, wall_size):
    try:
        wall_indeces = np.random.choice(len_wall_idxs, wall_size, replace=False)
        m = len_P - len_wall_idxs
        n = sample_size - wall_size
        #print(m, n, wall_size, m - n)
        indices = np.random.choice(m, n, replace=False) + len_wall_idxs
    except:
        raise Exception()
    return indices, wall_indeces
    
def normalize(X, axis=0):
    X -= np.mean(X, axis=axis)
    X /= np.std(X, axis=axis)
    return X
    
def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
    batch_size = states.shape[0]
    # generates random mini-batches until we have covered the full batch
    for _ in range(batch_size // mini_batch_size):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
        yield states[rand_ids], actions[rand_ids], log_probs[rand_ids], returns[rand_ids], advantage[rand_ids]
        
def compute_gae(value_t1, rewards, masks, values, gamma, lam):
    vs = np.zeros(values.shape[0] + 1)
    vs[:values.shape[0]] = values[:,0]
    vs[values.shape[0]] = value_t1
    gae = 0
    returns = []
    for step in reversed(range(rewards.shape[0])):
        delta = rewards[step] + gamma * vs[step + 1] * masks[step] - vs[step]
        gae = delta + gamma * lam * masks[step] * gae
        # prepend to get correct order back
        returns.insert(0, gae + vs[step])
    return np.array(returns)

def coordinate_system():
    line_set = o3d.geometry.LineSet()
    points = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]])
    colors = np.array([[1,0,0], [0,1,0], [0,0,1]])
    lines = np.array([[0,1], [0,2], [0,3]]).astype(int)
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    return line_set
    
def plot_voxels(voxels):
    vs = voxels.astype(np.int32)
    idxs = np.where(vs == 1)
    idxs_x = np_vec_to_mat(idxs[0])
    idxs_y = np_vec_to_mat(idxs[1])
    idxs_z = np_vec_to_mat(idxs[2])
    cat_indices = np.concatenate((idxs_x, idxs_y, idxs_z), axis=1)
    vox_pcd = o3d.geometry.PointCloud()
    vox_pcd.points = o3d.utility.Vector3dVector(cat_indices)
    o3d.visualization.draw_geometries([vox_pcd, coordinate_system()])
    
def estimate_normals_curvature(pc, nns, viewpoint = np.array([0,0,0]), k_neighbours = 30):
    m = pc.shape[0]
    assert(k_neighbours < m)
    n = pc.shape[1]
    k_nns = nns[:, 1:k_neighbours+1]
    p_nns = pc[k_nns[:]]
    
    p = np.matlib.repmat(pc, k_neighbours, 1)
    p = np.reshape(p, (m, k_neighbours, n))
    p = p - p_nns
    
    C = np.zeros((m,6))
    C[:,0] = np.sum(np.multiply(p[:,:,0], p[:,:,0]), axis=1)
    C[:,1] = np.sum(np.multiply(p[:,:,0], p[:,:,1]), axis=1)
    C[:,2] = np.sum(np.multiply(p[:,:,0], p[:,:,2]), axis=1)
    C[:,3] = np.sum(np.multiply(p[:,:,1], p[:,:,1]), axis=1)
    C[:,4] = np.sum(np.multiply(p[:,:,1], p[:,:,2]), axis=1)
    C[:,5] = np.sum(np.multiply(p[:,:,2], p[:,:,2]), axis=1)
    C /= k_neighbours
    
    normals = np.zeros((m,n))
    curvature = np.zeros((m,1))
    
    for i in range(m):
        C_mat = np.array([[C[i,0], C[i,1], C[i,2]],
            [C[i,1], C[i,3], C[i,4]],
            [C[i,2], C[i,4], C[i,5]]])
        values, vectors = np.linalg.eig(C_mat)
        lamda = np.min(values)
        k = np.argmin(values)
        normals[i,:] = vectors[:,k] / np.linalg.norm(vectors[:,k])
        curvature[i] = lamda / np.sum(values)
        
    pc_ = viewpoint - pc
    mask = np.sum(np.multiply(normals, pc_), axis=1) <= 0
    normals[mask,:] *= -1
    
    return normals, curvature
    
def hist(x, bins=50):
    plt.hist(x, bins=bins)
    plt.show()
    
def get_scene(nr, pc, label_col, max_K, normal_target=np.array([0., 0., 0.]), voxel_size=None, sample_size=None, voxel_mode = "None"):
    orig_labels, orig_indices, orig_counts = np.unique(pc[:,label_col], return_index = True, return_counts = True)
        
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc[:,:3])
    nns, flann = NN_o3d(pc=pc[:,:3], pcd=pcd, k_neighbours = max_K + 1)
    nns = nns.astype(int)
    _, curvature = estimate_normals_curvature(pc=pc[:,:3], nns=nns)
    if label_col == 3:
        knn_param = o3d.geometry.KDTreeSearchParamKNN(knn=30)
        pcd.estimate_normals(search_param=knn_param)
        pcd.orient_normals_towards_camera_location(camera_location=normal_target)
    else: # label_col = 6
        pcd.normals = o3d.utility.Vector3dVector(pc[:,3:6])
    pcd.normalize_normals()
    normals = np.asarray(pcd.normals)
    
    pc = np.hstack((pc[:,:3], normals, curvature, np_vec_to_mat(pc[:,label_col])))
    label_col = pc.shape[1] - 1
    
    wall_idxs = np.where(pc[:,label_col] == 0)[0]
    wall_idxs = np_vec_to_mat(wall_idxs)
    for i in range(1,6):
        wall_idxs = np.vstack((np_vec_to_mat(np.where(pc[:,label_col] == i)[0]), wall_idxs))
    if voxel_size is not None:
        scene = VoxScene(nr=nr,
            np_point_cloud=pc,
            o3d_pcd=pcd,
            nns=nns,
            flann=flann,
            wall_idxs=wall_idxs,
            label_col=label_col,
            orig_labels = orig_labels,
            orig_indices = orig_indices,
            orig_counts = orig_counts,
            voxel_size = voxel_size,
            sample_size=sample_size,
            voxel_mode=voxel_mode)
    else:
        scene = Scene(nr=nr,
            np_point_cloud=pc,
            o3d_pcd=pcd,
            nns=nns,
            flann=flann,
            wall_idxs=wall_idxs,
            label_col=label_col,
            orig_labels = orig_labels,
            orig_indices = orig_indices,
            orig_counts = orig_counts)
    return scene
    
    
def get_observation(scene, sample_size, wall_sample_size, max_steps_per_scene=15, point_mode="None"):
    if point_mode=="Voxel":
        return scene.voxels, None
    len_wall_idxs = scene.wall_idxs.shape[0]
    done = False
    wss = wall_sample_size
    while(not done):
        try:
            wall_size = int(sample_size * wss)
            indices, wall_indeces = get_indices(
                len_P=scene.np_point_cloud.shape[0], 
                sample_size=sample_size, 
                len_wall_idxs=len_wall_idxs, 
                wall_size=wall_size)
            done=True
        except:
            wss += 0.1
            if wss >= 1.0:
                raise Exception()

    obs_idxs = np.append(wall_indeces, indices, axis=0)
    np.random.shuffle(obs_idxs)
    
    obs = scene.np_point_cloud[obs_idxs,:scene.label_col]
    labels = -np.ones(obs.shape[0])
    obs = np.hstack((obs, np_vec_to_mat(labels)))
    if point_mode == "Query":
        querys = np.zeros((max_steps_per_scene, scene.np_point_cloud.shape[1]))
        obs = np.vstack((obs, querys))
    return obs, obs_idxs
    
def update_voxels(scene, labels, label_nr, voxel_mode="None"):
    new_labelled_idxs = np.where(labels == label_nr)[0]
    new_labelled_points = scene.np_point_cloud[new_labelled_idxs,:3]
    vn, vx, vy, vz = scene.voxelgrid.query(new_labelled_points)
    update = np.array(scene.voxels, copy=True)
    if voxel_mode=="None":
        update[vx,vy,vz]=0
    if voxel_mode=="Custom":
        update[vx,vy,vz,:]=np.zeros(5)
    return update
    
class Scene:
    def __init__(self, nr, np_point_cloud, o3d_pcd, nns, flann, wall_idxs, label_col, orig_labels, orig_indices, orig_counts):
        self.nr = nr
        self.np_point_cloud = np_point_cloud
        self.o3d_pcd = o3d_pcd
        self.nns = nns
        self.flann = flann
        self.wall_idxs = wall_idxs
        self.label_col = label_col
        self.orig_labels = orig_labels
        self.orig_indices = orig_indices
        self.orig_counts = orig_counts
        self.n_wall = np.sum(orig_counts[:6])
        self.n_objs = np.sum(orig_counts[6:])

class VoxScene(Scene):
    def __init__(self, nr, np_point_cloud, o3d_pcd, nns, flann, wall_idxs, label_col, orig_labels, orig_indices, orig_counts, voxel_size, sample_size, voxel_mode):
        super(VoxScene, self).__init__(nr, np_point_cloud, o3d_pcd, nns, flann, wall_idxs, label_col, orig_labels, orig_indices, orig_counts)
        if voxel_mode=="None":
            points = pd.DataFrame(self.np_point_cloud[:,:3])
            points.columns = ["x", "y", "z"]
            cloud = PyntCloud(points)
            
            voxelgrid_id = cloud.add_structure("voxelgrid", n_x=voxel_size, n_y=voxel_size, n_z=voxel_size)
            self.voxelgrid = cloud.structures[voxelgrid_id]
            self.voxels = self.voxelgrid.get_feature_vector(mode="binary")
        elif voxel_mode=="Custom":
            points = pd.DataFrame(self.np_point_cloud[:,:7])
            points.columns = ["x", "y", "z", "nx", "ny", "nz", "c"]
            cloud = PyntCloud(points)
            
            voxelgrid_id = cloud.add_structure("voxelgrid", n_x=voxel_size, n_y=voxel_size, n_z=voxel_size)
            self.voxelgrid = cloud.structures[voxelgrid_id]
            _, self.voxels = self.voxelgrid.get_feature_vector(mode="custom")