# hypernerf default 기반

# model params 추가
ModelParams = dict(
    scene =['MVI_1810'],
    video_length = 250,
    train_sample_rate = 1,
    test_sample_rate = 8,
    get_flame = True,
)

PipelineParams = dict(

)

ModelHiddenParams = dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 16,
     'resolution': [64, 64, 64, 150]
    },
    multires = [1,2,4,8],
    defor_depth = 2,
    net_width = 256,
    plane_tv_weight = 0.0002,
    time_smoothness_weight = 0.001,
    l1_time_planes =  0.001,
    # 추가
    flame_dims = [65, 30]

)
OptimizationParams = dict(
    # 추가
    max_gaussians = 1000000,

    dataloader=False,
    iterations = 60_000,
    # coarse 빼버림
    coarse_iterations = 0,
    densify_until_iter = 45_000,
    opacity_reset_interval = 6000,
    # position_lr_init = 0.00016,
    # position_lr_final = 0.0000016,
    # position_lr_delay_mult = 0.01,
    # position_lr_max_steps = 60_000,
    deformation_lr_init = 0.0016,
    deformation_lr_final = 0.00016,
    deformation_lr_delay_mult = 0.01,
    grid_lr_init = 0.016,
    grid_lr_final = 0.0016,
    # densify_until_iter = 50_000,
    opacity_threshold_coarse = 0.005,
    opacity_threshold_fine_init = 0.005,
    opacity_threshold_fine_after = 0.005,
    # pruning_interval = 2000
)