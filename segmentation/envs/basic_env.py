import gym
import numpy as np
import open3d as o3d
import math
import segmentation.envs.utils
import random

class BasicEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(
            self, 
            objs_dir="../point_cloud_env/objects",
            sample_size=1024, 
            wall_sample_size=0.3,
            unlabelled_punishment=1.0,
            false_labels_punishment=1.0,
            diff_punishment=0.05,
            max_steps_per_scene=20,
            debug=False,
            max_scenes=4,
            scene_mode="Random",
            training=True,
            point_mode="None",
            voxel_size=60,
            voxel_mode="None",
            single_scenes=False,
            early_diff=False,
            wall_weight=0.5
            ):
        super(BasicEnv, self).__init__()

        self.objs_dir=objs_dir
        self.debug = debug

        self.sample_size = sample_size
        self.wall_sample_size = wall_sample_size

        # punishment for unlabbeled points
        self.unlabelled_punishment = unlabelled_punishment
        self.false_labels_punishment = false_labels_punishment
        self.diff_punishment = diff_punishment
        self.scene_mode = scene_mode
        self.training=training
            
        self.max_steps_per_scene = max_steps_per_scene     

        # K nearest neighbours to grow a region
        self.min_K = 10
        self.max_K = 30

        self.min_room = -3.5
        self.max_room = 3.5

        self.min_params = np.array([self.min_room, self.min_room, self.min_room, self.min_K, (10/180)*math.pi, 0.025])
        self.max_params = np.array([self.max_room, self.max_room, self.max_room, self.max_K, 3*math.pi/8, 0.3])

        self.current_scene=0
        self.is_init=False
        self.max_scenes=max_scenes
        self.voxel_size=voxel_size
        self.voxel_mode=voxel_mode
        self.single_scenes=single_scenes
        self.early_diff = early_diff
        
        self.obj_weight = 1 - wall_weight
        self.wall_weight = wall_weight

        n_region_growing_params = self.min_params.shape[0]
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(n_region_growing_params,), dtype=np.float16)
        n_state_features = 8
        self.point_mode = point_mode
        if self.point_mode == "Voxel":
            if self.voxel_mode == "None":
                self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.voxel_size, self.voxel_size, self.voxel_size), dtype=np.float16)
            elif self.voxel_mode == "Custom":
                self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.voxel_size, self.voxel_size, self.voxel_size, 5), dtype=np.float16)
        elif self.point_mode == "Query":
            self.observation_space = gym.spaces.Box(low=-float("Inf"), high=float("Inf"), shape=(self.sample_size + self.max_steps_per_scene, n_state_features), dtype=np.float16)
        else: # None
            self.observation_space = gym.spaces.Box(low=-float("Inf"), high=float("Inf"), shape=(self.sample_size, n_state_features), dtype=np.float16)
        
        # generate some colors for the plot of the states
        self.colors = segmentation.envs.utils.generate_colors(max_colors=self.max_steps_per_scene)
        
        self.query_ps = []
        
        self.scene_idx = -1
        self.scenes = []
        self.total_reward_scene = 0

    def apply_action(self, action):
        action[3:] += 1
        action[3:] /= 2
        action[3:] *= self.max_params[3:]
        action[3] = max(self.min_params[3], action[3])
        action[4] = max(self.min_params[4], action[4])
        action[5] = max(self.min_params[5], action[5])
        query_p=action[:3] * self.max_params[:3]
        if self.debug: 
            print("Action: ", action)
            print("Query Point: ", query_p)
        self.query_ps.append(query_p)
        
        self.labels, no_candidate = segmentation.envs.utils.segment(all_points = self.scene.np_point_cloud, 
            flann=self.scene.flann,
            query_p=query_p, 
            labels = self.labels, 
            label_col = self.scene.label_col, 
            nns = self.scene.nns, 
            K=action[3], 
            angle_threshold=action[4], 
            curvature_threshold=action[5],
            region = self.region)
        return no_candidate

    def step(self, action):        
        change_scene = self.apply_action(action)
        
        if self.point_mode == "Voxel":
            self.obs = segmentation.envs.utils.update_voxels(scene=self.scene, labels=self.labels, label_nr=self.region, voxel_mode=self.voxel_mode)
        else:
            obs = self.obs[:self.sample_size, :]
            obs = np.hstack((obs[:, :self.scene.label_col], segmentation.envs.utils.np_vec_to_mat(self.labels[self.obs_idxs])))
            self.obs[:self.sample_size, :] = obs
            if self.point_mode == "Query":
                self.obs[self.sample_size + self.current_step - 1, :3] = self.query_ps[-1]
            labelled_idxs = np.where(self.obs[:,self.scene.label_col] != -1)[0]
            self.obs[labelled_idxs,self.scene.label_col] = 1
        
        self.region += 1
        n_points = self.scene.np_point_cloud.shape[0]
        n_errornous_points, n_unlabelled_points, diff, u_wall, u_obj, e_wall, e_obj = segmentation.envs.utils.compute_error(
            all_points = self.scene.np_point_cloud, 
            labels = self.labels, 
            label_col = self.scene.label_col, 
            orig_labels = self.scene.orig_labels, 
            orig_indices = self.scene.orig_indices, 
            orig_counts = self.scene.orig_counts)
        reward = segmentation.envs.utils.compute_weighted_reward(
            n_points=n_points, 
            e_wall=e_wall, u_wall=u_wall, n_wall=self.scene.n_wall, 
            e_obj=e_obj, u_obj=u_obj, n_objs=self.scene.n_objs, 
            wall_weight=self.wall_weight, obj_weight=self.obj_weight)
        if self.early_diff:
            n_objects = self.scene.orig_labels.shape[0]
            if self.region >= n_objects:
                reward -= diff * self.diff_punishment
        delta_reward = reward - self.last_reward
        #print(delta_reward)
        self.last_reward = reward
        
        self.current_step += 1
        if self.current_step >= self.max_steps_per_scene:
            change_scene = True
        self.total_reward_scene += delta_reward
        done = False
        if change_scene:
            if not self.early_diff:
                delta_reward -= diff * self.diff_punishment
                self.total_reward_scene += delta_reward 
            if not self.training:
                print("Total Reward in Scene ", self.current_scene, ":", self.total_reward_scene)
                self.render()
            if not self.single_scenes:
                done = self.scene_idx >= self.max_scenes - 1
                if not done:
                    self.next_scene()
            else: 
                done = True
        if not self.single_scenes:
            delta_reward /= self.max_scenes
        return self.obs, delta_reward, done, {"n_errornous_points/%": n_errornous_points/self.scene.np_point_cloud.shape[0], 
            "n_unlabelled/%": n_unlabelled_points/self.scene.np_point_cloud.shape[0], "current_scene": self.current_scene, "diff": diff}

    def reset(self):
        return self.next_observation()

    def render(self, mode='human', close=False):  
        segmentation.envs.utils.render_point_cloud(all_points = self.scene.np_point_cloud, 
            pcd = self.scene.o3d_pcd, 
            labels = self.labels, 
            colors = self.colors, 
            query_ps = self.query_ps)
            
    def render_state(self):
        if self.point_mode == "Voxel":
            segmentation.envs.utils.plot_voxels(self.obs)
            return
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.obs[:self.sample_size,:3])
        pcd.normals = o3d.utility.Vector3dVector(self.obs[:self.sample_size,3:6])
        segmentation.envs.utils.render_point_cloud(all_points = self.obs[:self.sample_size, :], 
            pcd = pcd, 
            labels = self.labels[self.obs_idxs], 
            colors = self.colors, 
            query_ps = self.query_ps)

    def get_current_scene(self):
        return self.current_scene
        
    def set_scene_order(self):
        self.scene_order = np.arange(self.max_scenes)
        if self.scene_mode == "Random":
            self.scene_order = np.array(random.sample(self.scene_order.tolist(), len(self.scene_order)))
        
    def next_scene(self):
        self.total_reward_scene = 0
        self.scene_idx += 1
        self.current_scene = self.scene_order[self.scene_idx]
        self.scene = self.scenes[self.current_scene]
        
        self.obs, self.obs_idxs = segmentation.envs.utils.get_observation(scene=self.scene, 
            sample_size=self.sample_size, 
            wall_sample_size=self.wall_sample_size, 
            max_steps_per_scene=self.max_steps_per_scene,
            point_mode=self.point_mode)

        self.max_params[:3] = np.abs(self.scene.np_point_cloud[:,:3]).max(axis=0)
        self.min_params[:3] = -self.max_params[:3]
        if self.point_mode=="None" or self.point_mode=="Query":
            self.obs[:,:3] = segmentation.envs.utils.normalize(self.obs[:,:3])    
            #print(np.mean(self.obs[:,6]))
            #print(np.std(self.obs[:,6]))
        self.region = 0
        self.labels = -np.ones(self.scene.np_point_cloud.shape[0])
        self.query_ps = []
        self.last_reward = 0
        self.current_step = 0

    # only called after scene is done
    def next_observation(self):
        if not self.is_init:
            for i in range(self.max_scenes):
                pc = np.loadtxt(self.objs_dir + "/PointcloudScenes/scene_" + str(i) + ".csv", delimiter=";")
                if self.point_mode == "Voxel":
                    scene = segmentation.envs.utils.get_scene(nr=i, pc=pc, label_col=6, max_K=30, voxel_size = self.voxel_size, sample_size=self.sample_size, voxel_mode=self.voxel_mode)
                else:
                    scene = segmentation.envs.utils.get_scene(nr=i, pc=pc, label_col=6, max_K=30)
                self.scenes.append(scene)
            self.is_init = True
            self.set_scene_order()
        
        if not self.single_scenes:
            self.set_scene_order()
            self.scene_idx = -1
        else: # self.single_scenes==True
            if self.scene_idx >= self.max_scenes - 1:
                self.set_scene_order()
                self.scene_idx = -1
                
        self.next_scene()
        return self.obs

    def close(self):
        return
