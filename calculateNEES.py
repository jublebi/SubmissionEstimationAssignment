import pandas as pd
import numpy as np
from bagpy import bagreader
import matplotlib.pyplot as plt

bag_path = 'ROVIO-Easy-WithCovariance.bag'
pose_topic = '/rovio/pose_with_covariance_stamped'
gt_topic = '/leica/position'
max_diff = 0.02

bag = bagreader(bag_path)
pose_csv = bag.message_by_topic(pose_topic)
leica_csv = bag.message_by_topic(gt_topic)

pose_df = pd.read_csv(pose_csv)
leica_df = pd.read_csv(leica_csv)

pose_df['timestamp'] = pose_df['header.stamp.secs'] + pose_df['header.stamp.nsecs'] * 1e-9
leica_df['timestamp'] = leica_df['header.stamp.secs'] + leica_df['header.stamp.nsecs'] * 1e-9

def extract_covariance_matrix(cov_string):
    try:
        cleaned = cov_string.replace('(', '').replace(')', '').replace('[', '').replace(']', '').replace('"', '')
        cov_list = list(map(float, cleaned.split(',')))
        return np.array(cov_list).reshape((6, 6))
    except Exception as e:
        print(f"Error parsing covariance: {e}")
        return np.full((6, 6), np.nan)

pose_df['cov_matrix'] = pose_df['pose.covariance'].apply(extract_covariance_matrix)

gt_times = leica_df['timestamp'].values
est_times = pose_df['timestamp'].values

matched_gt = []
valid_indices = []

j = 0
for i, t_est in enumerate(est_times):
    j = np.searchsorted(gt_times, t_est, side="left")
    if j == 0 or j >= len(gt_times):
        continue
    dt1 = abs(gt_times[j-1] - t_est)
    dt2 = abs(gt_times[j] - t_est)
    best_j = j-1 if dt1 < dt2 else j
    if min(dt1, dt2) <= max_diff:
        matched_gt.append(leica_df.iloc[best_j])
        valid_indices.append(i)

pose_synced = pose_df.iloc[valid_indices].reset_index(drop=True)
gt_synced = pd.DataFrame(matched_gt).reset_index(drop=True)

est_positions = pose_synced[['pose.pose.position.x', 'pose.pose.position.y', 'pose.pose.position.z']].to_numpy()
gt_positions = gt_synced[['point.x', 'point.y', 'point.z']].to_numpy()

#Got this alignment from the compareTrajectories script
R = np.array([
    [ 0.95851244,  0.28366682, -0.02805423],
    [-0.28358245,  0.95892212,  0.00702517],
    [ 0.02889463,  0.00122197,  0.99958172]
])
t = np.array([ 4.41075434, -1.92848381,  1.32647805 ])

est_aligned = (est_positions @ R.T) + t

nees_values = []
for i in range(len(est_aligned)):
    est = est_aligned[i]
    gt = gt_positions[i]
    P = pose_synced.iloc[i]['cov_matrix']
    if P.shape != (6, 6):
        nees_values.append(np.nan)
        continue
    P_pos = P[:3, :3]
    try:
        error = est - gt
        nees = error.T @ np.linalg.inv(P_pos) @ error
        nees_values.append(nees)
    except np.linalg.LinAlgError:
        nees_values.append(np.nan)

pose_synced['NEES'] = nees_values

pose_synced['t_rel'] = pose_synced['timestamp'] - pose_synced['timestamp'].iloc[0]

plt.figure(figsize=(10, 5))
plt.plot(pose_synced['t_rel'], pose_synced['NEES'], label='NEES', color='blue')
plt.xlabel('Time [s]')
plt.ylabel('NEES (3 DoF)')
plt.title('NEES over Time (aligned & time-synced)')
plt.legend()
plt.grid(True)
# plt.ylim(bottom=0, top=20)
plt.ylim(bottom=0)
plt.tight_layout()
plt.show()

mean_nees = np.nanmean(nees_values)
print(f"Average NEES (3 DoF): {mean_nees:.3f}")

pose_synced[['timestamp', 'NEES']].to_csv('nees_aligned_synced.csv', index=False)
