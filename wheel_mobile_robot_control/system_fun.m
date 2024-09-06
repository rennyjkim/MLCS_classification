function dY = system_fun(t, Y)
    dY = zeros(2,1);
    y = X(1);
    dy = X(2);
    e_psi = X(3)

    W_n = 1;
    V_x = 0.15;
    Kd = 2;
    Kp = 1;

    y_r = 0.05 * cos(t/pi);
    dy_r = -0.05 * (1/pi) * sin(t/pi);
    ddy_r = -0.05 * (1/pi)^2 * cos(t/pi);

    dY = zeros(3,1);
    dY(1) = dy;
    dY(2) = ddy_r - (Kd*(dy-dy_r)) - (Kp*(y-y_r));
    dY(3) = (1/V_x) * (-(Kd*(dy-dy_r)) - (Kp*(y-y_r)));
end