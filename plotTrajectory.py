from evo.core import sync, trajectory
from evo.tools import file_interface
from rosbags.rosbag1 import Reader as Rosbag1Reader
import matplotlib.pyplot as plt
import copy

# === Config ===
bag_path = "ROVIO-Easy-WithCovariance.bag"
gt_topic = "/leica/position"
est_topic = "/rovio/odometry"

# === Read bag and extract trajectories ===
with Rosbag1Reader(bag_path) as reader:
    traj_ref = file_interface.read_bag_trajectory(reader, gt_topic)
    traj_est = file_interface.read_bag_trajectory(reader, est_topic)

# === Synchronize trajectories (by timestamp) ===
traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)

print(f"Synced poses: {len(traj_ref.poses_se3)}")

traj_est_aligned = copy.deepcopy(traj_est)
traj_est_aligned.align(traj_ref, correct_scale=False, correct_only_scale=False)

# === Plot both trajectories ===
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)

ref_xyz = traj_ref.positions_xyz
est_xyz = traj_est_aligned.positions_xyz

ax.plot(ref_xyz[:, 0], ref_xyz[:, 1], label='Leica Ground Truth')
ax.plot(est_xyz[:, 0], est_xyz[:, 1], label='ROVIO Estimation', linestyle='--')
ax.set_title("Trajectory Comparison")
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.axis('equal')
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.show()
