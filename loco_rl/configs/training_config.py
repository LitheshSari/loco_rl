from legged_gym.envs.configs.basic_config import BasicCfg,BasicRunnerCfg
import numpy as np 
class RunnerCfg(BasicRunnerCfg):
    class algorithm(BasicRunnerCfg.algorithm):
        vae_learning_rate = 1e-3 
        kl_weight = 1.0
        
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        decoder_hidden_dims = [512, 256, 128]

        num_history = 5
        num_latent = 16 
        activation = 'elu'
    class runner:
        run_name = 'Test'
        experiment_name = 'DreamWaQ'
        
        num_steps_per_env = 24 # per iteration
        max_iterations = 1500 # number of policy updates
        # logging
        save_interval = 200 # check for potential saves every this many iterations
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt 

class EnvCfg(BasicCfg):
    class env(BasicCfg.env):
        num_envs = 4096
        num_observations = 45
        num_actions = 12
        num_observation_history = 5
        episode_length_s = 20  # episode length in seconds

        # ----------- Basic Observation ------------
        ## 所有 vel 都看
        observe_vel = False
        observe_only_ang_vel = True
        observe_contact_states = False 
        observe_command = True 

        ## 手动 gait 设定的 task, 必须 observe gait command, clock 和 giat indice 才有意义
        observe_two_prev_actions = False
        observe_gait_commands = False
        observe_timing_parameter = False
        observe_clock_inputs = False
        observe_imu = False

        # ---------- Privileged Observations ----------
        num_privileged_obs = 17 * 11 
        privileged_future_horizon = 1
        priv_observe_friction = False #! 1
        priv_observe_restitution = False #! 1 
        priv_observe_base_mass = False #! 1
        priv_observe_com_displacement = False #! 3
        priv_observe_motor_strength = False #! 12
        priv_observe_foot_height = False #! 4
        priv_observe_force_apply = False

        # ---------- Utils ----------
        need_other_obs_state = True
    class terrain(BasicCfg.terrain):   
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 5 # [m]
        
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = True
        # measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        # measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        measured_points_x = np.linspace(-0.8,0.8,17).tolist()
        measured_points_y = np.linspace(-0.5,0.5,11).tolist()
        curriculum = False # True
        selected = True # False # select a unique terrain type and pass all arguments
        
        terrain_kwargs = {
            "plane_terrain":{
                "weight": 1.0,
                "height" : 0.0
            },
            'random_uniform_terrain': {
                "weight": 10.0,
                "min_height" : -0.05,
                "max_height" : 0.05,
                "step" : 0.005,
                "downsampled_scale" : 0.2
            },
            # 'sloped_terrain': {
            #     "weight": 1.0,
            #     "slope" : 0.4
            # },
            'pyramid_sloped_terrain': {
                "weight": 1.0,
                "slope" : 0.4,
                'platform_size':3.0
            },
            'wave_terrain':{
                "weight": 1.0,
                'num_waves': 2,
                'amplitude': 0.5,
            },
            # 'stairs_terrain':{
            #     "weight": 1.0,
            #     'step_height': 0.15,
            #     'step_width': 0.5,
            # },
            'pyramid_stairs_terrain':{
                "weight": 1.0,
                'step_width': 0.5,
                'step_height': 0.15,
                'platform_size': 2.0,
            },
            'stepping_stones_terrain':{
                "weight": 1.0,
                'stone_size': 1.5,
                'stone_distance': 0.1,
                'max_height': 0.0,
                'platform_size': 4.,
                'depth': -10
            },
            'discrete_obstacles_terrain':{
                "weight": 1.0,
                "max_height":0.2, 
                "min_size" :1.0, 
                "max_size" : 2.0, 
                "num_rects" : 20, 
                "platform_size":1.
            },
            
        }
        
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 10 # number of terrain rows (levels)
        num_cols = 10 # number of terrain cols (types)

    class commands(BasicCfg.commands):
        command_curriculum = True
        cmd_cfg = {
            0:{
                'name':'vel_x',
                'init_low':-0.5,
                'init_high':0.5,
                'limit_low':-2.0,
                'limit_high':2.0,
                'local_range':0.5,
                'num_bins':21,
            },
            1:{
                'name':'vel_y',
                'init_low':-0.3,
                'init_high':0.3,
                'limit_low':-0.6,
                'limit_high':0.6,
                'local_range':0.5,
                'num_bins':11,
            },
            2:{
                'name':'vel_yaw',
                'init_low':-0.5,
                'init_high':0.5,
                'limit_low':-1.0,
                'limit_high':1.0,
                'local_range':0.5,
                'num_bins':11,
            }

        }
        
        num_commands = 3
        lin_vel_x = [-1.0, 1.0]  # min max [m/s]
        lin_vel_y =  [-0.6, 0.6]  # min max [m/s]
        ang_vel_yaw = [-1, 1]  # min max [rad/s]

        limit_vel_x = [-1.5, 1.5]
        limit_vel_y = [-0.6, 0.6]
        limit_vel_yaw = [-1.0, 1.0]
    class rewards(BasicCfg.rewards):
        only_positive_rewards = False  # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards_ji22_style = False
        sigma_rew_neg = 0.02

    class reward_scales:
        termination = -0.0
        tracking_lin_vel = 1.0
        tracking_ang_vel = 0.5

        lin_vel_z = -0.02
        ang_vel_xy = -0.001
        
        dof_vel = -1e-4
        dof_acc = -2.5e-7
        collision = -0.1
        action_rate = -0.001
        
        
        torques = -0.0001
        feet_slip = -0.01
        
        action_smoothness_1 = -0.01
        action_smoothness_2 = -0.01
        # raibert_heuristic = -10.0
        # feet_clearance_cmd_linear = -30

    class domain_rand(BasicCfg.domain_rand):
        rand_interval_s = 10
        randomize_rigids_after_start = True
        randomize_friction = True
        friction_range = [0.1, 3.0] # increase range
        randomize_restitution = True
        restitution_range = [0.0, 0.4]
        randomize_base_mass = True
        # add link masses, increase range, randomize inertia, randomize joint properties
        added_mass_range = [-1.0, 3.0]
        randomize_com_displacement = False
        # add link masses, increase range, randomize inertia, randomize joint properties
        com_displacement_range = [-0.15, 0.15]
        randomize_motor_strength = False
        motor_strength_range = [0.9, 1.1]
        randomize_lag_timesteps = True
        lag_timesteps = 6
        push_robots = False
        push_interval_s = 15
        max_push_vel_xy = 1.




class PlayEnvCfg(EnvCfg):
    class env(EnvCfg.env):
        num_envs = 50
    
    class terrain(EnvCfg.terrain):   
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh

        terrain_kwargs = {'type':"random_uniform_terrain",
                          'min_height':-0.05,
                          'max_height':0.05,
                          'step':0.005} # None # Dict of arguments for selected terrain
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 2 # number of terrain rows (levels)
        num_cols = 2 # number of terrain cols (types)

    class domain_rand(EnvCfg.domain_rand):
        # add link masses, increase range, randomize inertia, randomize joint properties
        randomize_rigids_after_start = False
        randomize_friction = False
        randomize_restitution = False
        randomize_base_mass = False
        # add link masses, increase range, randomize inertia, randomize joint properties
        randomize_com_displacement = False
        # add link masses, increase range, randomize inertia, randomize joint properties
        randomize_motor_strength = False
        randomize_Kp_factor = False
        randomize_Kd_factor = False
        randomize_gravity = False
        push_robots = False
        randomize_lag_timesteps = False
    