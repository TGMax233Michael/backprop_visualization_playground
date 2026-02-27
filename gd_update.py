from gd_algo import sgd, momentum, adagrad, rmsprop, adam

def update(start_points, optimize_target, grad_updater_type, lr, beta_m=0.9, beta_v=0.9, epoches=500):
    loss_history = []
    pos_history = []
    m = 0.0
    V = 0.0
    pos = start_points
    for e in range(epoches):
        loss = optimize_target.func(pos)
        grad = optimize_target.grad(pos)
        loss_history.append(loss)
        pos_history.append(pos)
        if grad_updater_type == "sgd":
            pos = sgd(grad, pos, lr)
        elif grad_updater_type == "momentum":
            pos, m = momentum(grad, pos, lr, beta_m, m)
        elif grad_updater_type == "adagrad":
            pos, V = adagrad(grad, pos, lr, V, 1e-8)
        elif grad_updater_type == "rmsprop":
            pos, V = rmsprop(grad, pos, lr, V, 1e-8, beta_v)
        elif grad_updater_type == "adam":
            pos, V, m = adam(grad, pos, lr, V, 1e-8, beta_m, beta_v, m)
        else:
            raise TypeError(f"no updater type {grad_updater_type}")
    return pos_history, loss_history