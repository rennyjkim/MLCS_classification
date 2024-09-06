clear;
close all;
clc

V_x = 0.15;
R_wheel = 0.03;
R_robot = 0.075;
Kd = 2;
Kp = 1;

init = [1,0,0];
[t,X] = ode45(@(t,X) system_fun(t,X), [0:0.1:50], init,'.');

y_r = 0.05 * cos(t/pi);
dy_r = -0.05 * sin(t/pi);

w = (-Kd * (X(:,2) - dy_r) - Kp * (X(:,1) - y_r)) / V_x;
W_R = R_robot * w / (2 * R_wheel) + V_x / R_wheel;
W_L = 2 * V_x / R_wheel - W_R;

subplot(3,1,1);
plot(t,X(:,1),t,y_r)
title('y, y_r')
legend('y', 'y_r')
xlabel('time(s)')
ylabel('y(m)')

subplot(3,1,2);
plot(t, X(:, 3))
title('e_{psi}')
legend('e_{psi}')
xlabel('time(s)')
ylabel('e_{psi}(rad)')

subplot(3,1,3);
plot(t,W_R,t,W_L);
title('W_R, W_L')
legend('W_R', 'W_L');
xlabel('time(s)');
ylabel('W_R, W_L(rad/s)')