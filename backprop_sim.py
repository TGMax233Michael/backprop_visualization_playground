import numpy as np
from argparse import ArgumentParser
from datetime import datetime

from analysis import sim
from optim_target import Optimize_Target

optim_target_registry = {cls.__name__.lower(): cls for cls in Optimize_Target.__subclasses__()}

def get_parser():
    parser = ArgumentParser(description="used for Gradient Decendent Algorithm simulation and visualization")
    parser.add_argument("-lr", default=0.01, type=float)
    parser.add_argument("-epochs", default=500, type=int)
    parser.add_argument("-f", default="simple", type=str)
    parser.add_argument("-html", default=False, type=bool, help="Set True to save matplotlib animation as html file")
    parser.add_argument("-start", nargs=2, default=None, type=float, help="Start Position for simulation, e.g. (x, y)")
    return parser

def get_optim_target(name):
    name = name.lower()
    optim_target = optim_target_registry.get(name)
    if optim_target is None:
        raise ValueError(f"Unknown Optimization Target Name, currently support {list(optim_target_registry.keys())}")
    return optim_target()

def get_file_name():
    cur_time = datetime.now().strftime("%Y%m%d%H%M%S")
    file_name = cur_time + "_ani.html"
    return file_name
    
def get_proper_plotting_range(start_pos):
    x, y = start_pos
    if x > 0:
        x_range = (-x*0.5, x*1.5)
    else:
        x_range = (x*1.5, -x*0.5)
    if y > 0:
        y_range = (-y*0.5, y*1.5)
    else:
        y_range = (y*1.5, -y*0.5)
    return x_range, y_range
        
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    lr = args.lr
    epochs = args.epochs
    target_name = args.f
    to_save_html = args.html
    start_pos = args.start
    optim_target = get_optim_target(target_name)
    file_name = get_file_name()
    
    if start_pos is None:
        start_pos = optim_target.recommend_start_pos
    else:
        start_pos = np.array(start_pos)
        optim_target.check_start_pos(start_pos)
    x_range, y_range = get_proper_plotting_range(start_pos)
    print(x_range)
    print(y_range)
    
    sim(optim_target, epochs, lr, start_pos, x_range, y_range, 256, is_save=to_save_html, file_name=file_name)