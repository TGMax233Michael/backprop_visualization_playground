import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
import numpy as np
import os

from utils import is_contain_nan

plt.rcParams['animation.embed_limit'] = 200.0

def init_fig():
    fig = plt.figure(dpi=72, figsize=(22, 7))
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)
    return fig, ax1, ax2, ax3

def draw_func_3D(ax1, optimize_target, x_range, y_range, num=100):
    x_min, x_max = x_range
    y_min, y_max = y_range
    X, Y = np.meshgrid(np.linspace(x_min, x_max, num), np.linspace(y_min, y_max, num))
    grid = np.stack([X, Y], axis=-1)
    Z = optimize_target.func(grid)
    func_3D = ax1.plot_surface(X, Y, Z, cmap=cm.coolwarm)
    return ax1, func_3D

def draw_func_contour(ax2, optimize_target, x_range, y_range, num=100):
    x_min, x_max = x_range
    y_min, y_max = y_range
    X, Y = np.meshgrid(np.linspace(x_min, x_max, num), np.linspace(y_min, y_max, num))
    grid = np.stack([X, Y], axis=-1)
    Z = optimize_target.func(grid)
    func_contour = ax2.contourf(X, Y, Z, cmap=cm.coolwarm)
    return ax2, func_contour

# def draw_trace_3D(ax1, pos_history, loss_history, label, color):
#     pos_history = np.array(pos_history)
#     x_history = pos_history[:, 0]
#     y_history = pos_history[:, 1]
#     ax1.plot3D(x_history, y_history, loss_history, zorder=10, c=color, label=label)
#     return ax1

# def draw_trace_contour(ax2, pos_history, label, color):
#     pos_history = np.array(pos_history)
#     x_history = pos_history[:, 0]
#     y_history = pos_history[:, 1]
#     ax2.plot(x_history, y_history, c=color, label=label)
#     return ax2

# def draw_start_end_3D(ax1, start, end, color):
#     x_start, y_start, z_start = start
#     x_end, y_end, z_end = end
#     ax1.scatter([x_start], [y_start], [z_start], zorder=10, c=color, marker="x")
#     ax1.scatter([x_end], [y_end], [z_end], zorder=10, c=color, marker="*")
#     return ax1

# def draw_start_end_contour(ax2, start, end, color):
#     x_start, y_start, z_start = start
#     x_end, y_end, z_end = end
#     ax2.scatter([x_start], [y_start], c=color, marker="x")
#     ax2.scatter([x_end], [y_end], c=color, marker="*")
#     return ax2

def draw_trace(optimize_target, history_dict, start_pos, lr, epoches, x_range, y_range, num=100, colors=None, is_save=False, file_name=None):
    fig, ax1, ax2, ax3 = init_fig()
    n_methods = len(list(history_dict.keys()))
    *optimal_pos, optimal_loss = optimize_target.optimal_res
    start_pos_loss = optimize_target.func(start_pos)
    loss_x = np.arange(0, epoches)

    ax1, func_3D = draw_func_3D(ax1, optimize_target, x_range, y_range, num)
    ax2, func_contour = draw_func_contour(ax2, optimize_target, x_range, y_range, num)
    ax1.scatter(*optimal_pos, optimal_loss, marker="X", c="lime", s=96, zorder=10)
    ax1.text(*optimal_pos, optimal_loss, f"{optimal_pos[0]:.3f}, {optimal_pos[1]:.3f}")
    ax2.scatter(*optimal_pos, marker="X", c="lime", s=48, label="Optimal")
    ax2.text(*optimal_pos, f"{optimal_pos[0]:.3f}, {optimal_pos[1]:.3f}", rotation=45)
    ax3.axhline(y=optimal_loss, color="Black", linestyle="dashed", label="Optimal Loss")
    ax3.text((epoches+1)/2, optimal_loss - 0.05* start_pos_loss, s=f"f({optimal_pos[0]:.3f}, {optimal_pos[1]:.3f}) = {optimal_loss:.3f}")
    ax3.set_xlim(0, epoches*1.1)
    ax3.set_ylim(optimal_loss - 0.1*np.abs(start_pos_loss), start_pos_loss + 0.2*np.abs(start_pos_loss))

    if colors is None:
        colors = cm.rainbow(np.linspace(0, 1, n_methods))

    plt.suptitle(f"Target: {optimize_target.func_expression}\nlr={lr} epoches={epoches}")

    _3d_lines = []
    _3d_scatters = []
    contour_lines = []
    contour_scatters = []
    loss_lines = []
    to_draws = []

    for i, (k, method_history) in enumerate(history_dict.items()):
        pos_history, loss_history = method_history
        end_pos = (pos_history[-1][0], pos_history[-1][1], loss_history[-1])
        ax2.scatter(start_pos[0], start_pos[1], marker=".", s=32, color=colors[i])
        if is_contain_nan(end_pos):
            print(f"Method [{k}] result data contains nan, no data ploting!")
            line_3d = ax1.plot(start_pos[0], start_pos[1], loss_history[0], color=colors[i], zorder=10)
            scatter_3d = ax1.scatter(start_pos[0], start_pos[1], loss_history[0], color=colors[i], marker="*", s=64, zorder=10)
            line = ax2.plot(start_pos[0], start_pos[1], label=f"{k}: Failed", color=colors[i])
            scatter = ax2.scatter(start_pos[0], start_pos[1], color=colors[i], marker="*", s=64, zorder=10)
            line_loss = ax3.plot(0, loss_history[0], label=f"{k}: Failed", color=colors[i])
            to_draws.append(False)
        else:
            line_3d = ax1.plot(start_pos[0], start_pos[1], loss_history[0], color=colors[i], linestyle="--", linewidth=2)
            scatter_3d = ax1.scatter(start_pos[0], start_pos[1], loss_history[0], color=colors[i], marker="*", s=64, zorder=10)
            line = ax2.plot(start_pos[0], start_pos[1], label=f"{k}", color=colors[i], linestyle="--", linewidth=2)
            scatter = ax2.scatter(start_pos[0], start_pos[1], color=colors[i], marker="*", s=64, zorder=10)
            line_loss = ax3.plot(0, loss_history[0], label=f"{k}", color=colors[i], linewidth=2)
            to_draws.append(True)
        _3d_lines.append(line_3d)
        _3d_scatters.append(scatter_3d)
        contour_lines.append(line)
        contour_scatters.append(scatter)
        loss_lines.append(line_loss)

    contour_title = ax2.text(0.5, 0.95, "Epoch: 0", transform=ax2.transAxes, ha="center", fontsize=12, animated=True, c="black", clip_on=False)
    ax3.legend()
    fig.colorbar(func_contour, ax=ax2, shrink=0.9)

    def update(frame):
        artists = []
        current_idx = frame+1
        contour_title.set_text(f"Epoch: {current_idx}")

        for i, (k, method_history) in enumerate(history_dict.items()):
            if to_draws[i] is True:
                pos_history, loss_history = method_history
                pos_history = np.array(pos_history)
                x = pos_history[:current_idx, 0]
                y = pos_history[:current_idx, 1]
                z = loss_history[:current_idx]
                _3d_lines[i][0].set_data(x, y)
                _3d_lines[i][0].set_3d_properties(z)
                contour_lines[i][0].set_data(x, y)
                loss_lines[i][0].set_data(loss_x[:current_idx], z)
                if len(x) >= 1:
                    contour_scatters[i].set_offsets([x[-1], y[-1]])
                    # print(_3d_scatters[i]._offsets3d)
                    # _3d_scatters[i]._offsets3d = (np.float64(x[-1]), np.float64(y[-1]), np.float64(z[-1]))
            artists.append(_3d_lines[i][0])
            artists.append(_3d_scatters[i])
            artists.append(contour_lines[i][0])
            artists.append(loss_lines[i][0])
            artists.append(contour_scatters[i])
        artists.append(contour_title)
        return artists
    
    ani = animation.FuncAnimation(fig=fig, func=update, frames=epoches, interval=5, blit=True)
    
    plt.grid(True)
    plt.show()

    if is_save is True:
        if file_name is None:
            file_name = "animation.mp4 "
        if not os.path.exists("./results"):
            try:
                os.mkdir("./results")
            except OSError as e:
                raise ValueError(f"Failed to create directory ./results: {e}")
        writer = animation.FFMpegFileWriter(fps=30)
        ani.save("./results/"+file_name, writer=writer)

    return ani

