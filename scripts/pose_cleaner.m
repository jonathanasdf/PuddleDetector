% matlab script to load pose data and convert to homogeneous matrices
poses = csvread('../data/odom.log'); % be sure to manually delete the first line of odom.log that is a text comment
                                     % I don't know why matlab's csv read is so stupid that it cannot deal with comments
timestamps = poses(:,1) * 1e-9; % nanoseconds
translations = poses(:, 6:8);
quaternions_xyzw = poses(:, 9:12);

T1 = inv([quat2rotm(quaternions_xyzw(1, :)) translations(1, :)'; 0 0 0 1]);
T = [];
for i = 1:size(times)
    T = [T; timestamps(i) reshape((T1 * [quat2rotm(quaternions_xyzw(i, :)) translations(i, :)'; 0 0 0 1])', 1, [])];
end
dlmwrite('../data/odom_clean.dat', T, 'delimiter', ' ', 'precision', 20);
