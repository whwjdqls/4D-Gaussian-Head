def merge_hparams(args, config):
    params = ["OptimizationParams", "ModelHiddenParams", "ModelParams", "PipelineParams"]
    for param in params:
        if param in config.keys():
            for key, value in config[param].items():
                if hasattr(args, key):
                    setattr(args, key, value)
                else: # 수정, key 없어도 value 추가!
                    setattr(args, key, value)

    return args

# Add key's from config to params
def cfg2params(params,config):
    for cfg_key in config.keys():
        setattr(params, cfg_key, config[cfg_key])
    return params