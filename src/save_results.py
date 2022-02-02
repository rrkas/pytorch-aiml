import os
import datetime

results_dir_name = "results"
os.makedirs(results_dir_name, exist_ok=True)


def save_plt(plt, plot_name=None):
    now = datetime.datetime.now()
    file_prefix = f"{now.year:04d}{now.month:02d}{now.day:02d}{now.hour:02d}{now.minute:02d}{now.second:02d}{now.microsecond:06d}"
    if plot_name is None:
        plot_name = ""
    else:
        plot_name = f"_{plot_name}_"
    file_name = file_prefix + plot_name + ".png"
    file_path = os.path.join(results_dir_name, file_name)
    c = 0
    while os.path.exists(file_path):
        file_name = file_prefix + plot_name + f"_{c}.png"
        c += 1
    file_path = os.path.join(results_dir_name, file_name)
    plt.savefig(file_path, bbox_inches="tight")
