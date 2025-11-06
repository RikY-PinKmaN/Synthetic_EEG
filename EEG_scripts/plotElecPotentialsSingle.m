function plotElecPotentialsSingle(EMap, potentials)
% =========================================================================
% MODIFIED aEEG_TOOLBOX plotting function to work with subplots.
% Original by: Fabien Lotte, Ang Kai Keng
% Modifications:
% - Removed figure and subplot creation to allow plotting on existing axes.
% - Removed hardcoded titles and labels.
% =========================================================================

% --- Define Channels and Extract Coordinates ---
channels = [3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 15, 17];
X = cell2mat(EMap(2,channels));
Y = cell2mat(EMap(3,channels));
Z1 = potentials';

% --- Head, Nose and Ear Constants for the Plot ---
rmax     = 1.0;      % Head radius
CIRCGRID = 201;      % Number of angles to use in drawing circles
hwidth   = 0.007;    % Width of head ring
hin      = rmax * (1 - hwidth / 2); % Inner head ring radius
base     = rmax - 0.0046;
basex    = 0.18 * rmax; % Nose width
tip      = 1.15 * rmax;
tiphw    = 0.04 * rmax; % Nose tip half width
tipr     = 0.01 * rmax; % Nose tip rounding
q        = 0.04;      % Ear length
EarX     = [.497-.005  .510  .518  .5299 .5419  .54    .547   .532   .510   .489-.005] * 2;
EarY     = [q+.0555 q+.0775 q+.0783 q+.0746 q+.0555 -.0055 -.0932 -.1313 -.1384 -.1199] * 2;

% --- Interpolate Data for a Smooth Surface Plot ---
GRID_SCALE = 200; % Plot map on a 200x200 grid
iiY = linspace(min(X), max(X), GRID_SCALE); % x-axis for interpolation
iiX = linspace(min(Y), max(Y), GRID_SCALE); % y-axis for interpolation
[iX, iY, iiZ1] = griddata(X, Y, Z1, iiY', iiX, 'linear'); % Interpolate data

% --- Mask Points Outside the Head ---
% This prevents the contour from extending into a square shape
cnv = convhull(X, Y);
maskConvHull = inpolygon(iX, iY, X(cnv), Y(cnv));
iiZ1(maskConvHull == 0) = NaN;

% --- PLOTTING ---
% The following commands will draw on the CURRENTLY ACTIVE axes, which is
% controlled by the `subplot` command in your main script.

% 1. Plot the interpolated potentials as a filled contour map
contourf(iX, iY, iiZ1, 20, 'LineColor', 'None');
hold on;

% 2. Plot the Head Outline, Nose, and Ears
% Plot head
circ = linspace(0, 2 * pi, CIRCGRID);
rx = sin(circ);
ry = cos(circ);
headx = [[rx(:)' rx(1)] * (hin + hwidth) [rx(:)' rx(1)] * hin];
heady = [[ry(:)' ry(1)] * (hin + hwidth) [ry(:)' ry(1)] * hin];
patch(headx, heady, ones(size(headx)), 'k', 'edgecolor', 'k');

% Plot nose
plot([basex; tiphw; 0; -tiphw; -basex], [base; tip-tipr; tip; tip-tipr; base], ...
     'Color', 'k', 'LineWidth', 2);

% Plot ears
plot(EarX, EarY, 'color', 'k', 'LineWidth', 2);
plot(-EarX, EarY, 'color', 'k', 'LineWidth', 2);

% 3. Plot the Electrode Locations
plot(X, Y, 'o', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'k', 'MarkerSize', 2);

% 4. Final Axis Adjustments
axis square;
xlim([-1.2 1.2]);
ylim([-1 1.2]);
axis off;
hold off;

% 5. Plot the Colorbar
colorbar;

end