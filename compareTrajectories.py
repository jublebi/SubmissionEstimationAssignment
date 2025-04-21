from evo.tools import log
from evo.tools import file_interface
from evo.tools import plot
import pprint
from rosbags.rosbag1 import Reader as Rosbag1Reader
import numpy as np
import matplotlib.pyplot as plt
from yaml.reader import Reader
from evo.core import sync
from evo.core import metrics
from evo.core.units import Unit
import copy

log.configure_logging(verbose=True, debug=True, silent=False)


# temporarily override some package settings
from evo.tools.settings import SETTINGS
SETTINGS.plot_usetex = False

plot.apply_settings(SETTINGS)

# === Config ===
bag_path = "ROVIO-Easy-WithCovariance.bag"
gt_topic = "/leica/position"
est_topic = "/rovio/odometry"
est_topicVins = "/vins_estimator/odometry"
first_monocular_bag_path = "VinsMonoEurocHard.bag"

# === Read bag and extract trajectories ===
with Rosbag1Reader(bag_path) as reader:
    traj_ref = file_interface.read_bag_trajectory(reader, gt_topic)
    traj_est = file_interface.read_bag_trajectory(reader, est_topic)
max_diff = 0.02

with Rosbag1Reader(first_monocular_bag_path) as reader2:
    traj_ref2 = file_interface.read_bag_trajectory(reader2, gt_topic)
    traj_est2 = file_interface.read_bag_trajectory(reader2, est_topicVins)

traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est, max_diff)
traj_ref2, traj_est2 = sync.associate_trajectories(traj_ref2, traj_est2, max_diff)

traj_est_aligned = copy.deepcopy(traj_est)
traj_est_aligned.align(traj_ref, correct_scale=False, correct_only_scale=False)

traj_est_aligned2 = copy.deepcopy(traj_est2)
traj_est_aligned2.align(traj_ref2, correct_scale=False, correct_only_scale=False)

traj_by_label = {
  #  "estimate ROVIO (not aligned)": traj_est,
    "estimate ROVIO (aligned)": traj_est_aligned,
  #  "estimate VINSMono (not aligned)": traj_est2,
    "estimate VINSMono (aligned)": traj_est_aligned2,
    "reference": traj_ref
}
fig = plt.figure()
plot.trajectories(fig, traj_by_label, plot.PlotMode.xyz)
plt.show()

pose_relation = metrics.PoseRelation.translation_part
use_aligned_trajectories = True

if use_aligned_trajectories:
    data = (traj_ref, traj_est_aligned)
else:
    data = (traj_ref, traj_est)

ape_metric = metrics.APE(pose_relation)
ape_metric.process_data(data)

ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse)
print(ape_stat)

ape_stats = ape_metric.get_all_statistics()
pprint.pprint(ape_stats)

seconds_from_start = [t - traj_est.timestamps[0] for t in traj_est.timestamps]
fig = plt.figure()
plot.error_array(fig.gca(), ape_metric.error, x_array=seconds_from_start,
                 statistics={s:v for s,v in ape_stats.items() if s != "sse"},
                 name="APE", title="APE w.r.t. " + ape_metric.pose_relation.value, xlabel="$t$ (s)")
plt.show()

plot_mode = plot.PlotMode.xy
fig = plt.figure()
ax = plot.prepare_axis(fig, plot_mode)
plot.traj(ax, plot_mode, traj_ref, '--', "gray", "reference")
plot.traj_colormap(ax, traj_est_aligned if use_aligned_trajectories else traj_est, ape_metric.error,
                   plot_mode, min_map=ape_stats["min"], max_map=ape_stats["max"])
ax.legend()
plt.show()