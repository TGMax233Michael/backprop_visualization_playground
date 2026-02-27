import numpy as np

from gd_update import update
from visual import draw_trace
from utils import is_contain_nan

def analyse(update_type, pos_history, loss_history, lr):
    epoches = len(pos_history)
    print(f"{update_type.upper():-^30}")
    print(f"Learning Rate: {lr}")
    print("Head 5 data:")
    print(loss_history[0])
    for i in range(5):
        print(f"{'Epoch'+str(i+1):<10} | pos: {np.array2string(pos_history[i], precision=5, separator=',')} | loss: {loss_history[i]:.5e}")
    print("Tail 5 data:")
    for i in range(epoches-5,epoches):
        print(f"{'Epoch'+str(i+1):<10} | pos: {np.array2string(pos_history[i], precision=5, separator=',')} | loss: {loss_history[i]:.5e}")

def process_plotting_range(x_range, y_range, optimal_pos, history_dict):
    # check optim pos in range
    x_opt, y_opt = optimal_pos
    x_min, x_max = x_range
    y_min, y_max = y_range
    x_min, x_max = min(x_opt*1.5, x_min), max(x_opt*1.5, x_max)
    y_min, y_max = min(y_opt*1.5, y_min), max(y_opt*1.5, y_max)
    # check runtime extremum in range
    extremums = np.zeros((len(list(history_dict.keys())), 4))
    """
    [
        [x_min, x_max, y_min, y_max],
        ...
    ] (n_methdos, 4)
    """
    for i, (pos_history, _) in enumerate(history_dict.values()):
        pos_history = np.array(pos_history)
        x_coords, y_coords = pos_history[:, 0], pos_history[:, 1]
        x_coord_min, x_coord_max = np.min(x_coords), np.max(x_coords)
        y_coord_min, y_coord_max = np.min(y_coords), np.max(y_coords)
        extremums[i] = np.array([x_coord_min, x_coord_max, y_coord_min, y_coord_max])
    x_ex_min, x_ex_max = np.nanmin(extremums[:, 0]), np.nanmax(extremums[:, 1])
    y_ex_min, y_ex_max = np.nanmin(extremums[:, 2]), np.nanmax(extremums[:, 3])
    x_min, x_max = min(x_ex_min*1.5, x_min), max(x_ex_max*1.5, x_max)
    y_min, y_max = min(y_ex_min*1.5, y_min), max(y_ex_max*1.5, y_max)
    return (x_min, x_max), (y_min, y_max)

def sim(optimize_target, epoches, lr, start_point, x_range, y_range, num=100, is_save=False, file_name=None):
    sgd_pos_history, sgd_loss_history = update(start_point, optimize_target, "sgd", lr=lr, epoches=epoches)
    momentum_pos_history, momentum_loss_history = update(start_point, optimize_target, "momentum", lr=lr, epoches=epoches)
    adagrad_pos_history, adagrad_loss_history = update(start_point, optimize_target, "adagrad", lr=lr, epoches=epoches)
    rmsprop_pos_history, rmsprop_loss_history = update(start_point, optimize_target, "rmsprop", lr=lr, epoches=epoches)
    adam_pos_history, adam_loss_history = update(start_point, optimize_target, "adam", lr=lr, epoches=epoches)
    analyse("sgd", sgd_pos_history, sgd_loss_history, lr)
    analyse("momentum", momentum_pos_history, momentum_loss_history, lr)
    analyse("adagrad", adagrad_pos_history, adagrad_loss_history, lr)
    analyse("rmsprop", rmsprop_pos_history, rmsprop_loss_history, lr)
    analyse("adam", adam_pos_history, adam_loss_history, lr)
    
    all_final_res = [
        ("sgd", sgd_loss_history[-1]),
        ("momentum", momentum_loss_history[-1]),
        ("adagrad", adagrad_loss_history[-1]),
        ("rmsprop", rmsprop_loss_history[-1]),
        ("adam", adam_loss_history[-1])
    ]
    all_final_res.sort(key=lambda x: x[1] if not np.isnan(x[1]) else float("inf"))
    print(f"{'Rank':-^30}")
    for name, res in all_final_res:
        print(f"{name:-<10}: {res:.5g}")
    print("-"*30)

    history_dict = {
        "sgd": (sgd_pos_history, sgd_loss_history),
        "momentum": (momentum_pos_history, momentum_loss_history),
        "adagrad": (adagrad_pos_history, adagrad_loss_history),
        "rmsprop": (rmsprop_pos_history, rmsprop_loss_history),
        "adam": (adam_pos_history, adam_loss_history)
    }
    
    x_range, y_range = process_plotting_range(x_range, y_range, optimize_target.optimal_res[:2], history_dict)
    draw_trace(optimize_target, history_dict, start_point, lr, epoches, x_range, y_range, num, is_save=is_save, file_name=file_name)