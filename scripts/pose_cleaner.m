% matlab script to load pose data and convert to homogeneous matrices
poses = csvread('../data/odom.log'); % be sure to manually delete the first line of odom.log that is a text comment
                                     % I don't know why matlab's csv read is so stupid that it cannot deal with comments
timestamps = poses(:,1) * 1e-9; % nanoseconds
timestamps_interp = [];
translations = poses(:, 6:8);
translations = bsxfun(@minus, translations, translations(1,:));
quaternions_xyzw = poses(:, 9:12);

T1 = inv([quat2rotm(quaternions_xyzw(1, :)) translations(1, :)'; 0 0 0 1]);
T = [];
jumps = [];
TT = {};
k = 1;
gaps = [];
for i = 1:size(timestamps)
    if i > 2 && timestamps(i) - timestamps(i-1) > 0.015
        timestamps_interp = [timestamps_interp timestamps(i-1) + timestamps(i-1) - timestamps(i-2)];
        TT{k} = TT{k-1} * inv(TT{k-2}) * TT{k-1};
        gaps = [gaps; TT{k}(1,4) TT{k}(2,4) TT{k}(3,4)];
        k = k + 1;
    end
    timestamps_interp = [timestamps_interp timestamps(i)];
    TT{k} = T1 * [quat2rotm(quaternions_xyzw(i, :)) translations(i, :)'; 0 0 0 1];
    k = k+1;
end
k = k-1;
for i = 2:k
    dT = TT{i} * inv(TT{i-1});
    dt = norm(dT(1:3, 4));
    if dt > 0.07
        jumps = [jumps i];
    end
end
for i = 1:k
    T = [T; timestamps_interp(i) reshape(TT{i}', 1, [])];
end
dlmwrite('../data/odom_clean.dat', T, 'delimiter', ' ', 'precision', 20);
hold off; plot3(T(:,5), T(:,9), T(:,13), '.');
hold on; plot3(T(jumps,5), T(jumps,9), T(jumps,13), 'ro', 'MarkerSize', 3);
hold on; plot3(T(jumps-1,5), T(jumps-1,9), T(jumps-1,13), 'go', 'MarkerSize', 3);
hold on; plot3(gaps(:,1), gaps(:,2), gaps(:,3), 'ko', 'MarkerSize', 3);
