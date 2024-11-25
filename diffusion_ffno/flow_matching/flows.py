def optimal_transport(target_1, target_0, t, flow_params):
    sigma_min, = flow_params
    input_ = (1 - (1 - sigma_min)*t)*target_0 + t*target_1
    output_ = target_1 - (1 - sigma_min)*target_0
    return input_, output_