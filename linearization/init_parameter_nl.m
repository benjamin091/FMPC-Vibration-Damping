function parameter=init_parameter_nl()

%% Abmessungen
parameter.g_RS = 2.6;       % = r_Ring/r_sun = r_3/r_2
parameter.r_3  = 0.3;     % r_Ring [m]
parameter.r_2 = parameter.r_3 / parameter.g_RS;    % sun wheel
parameter.r_1 = (parameter.r_2 +parameter.r_3)/2; % carrier
parameter.r_p = parameter.r_3-parameter.r_2;

parameter.g_TT = 66/54*80/23; % DriveTrain Transmission ratio
parameter.r_Wheel=0.3; 

%% Feder-Dämpfer-Terme
% parameter.k_3  = 10;     % stiffness ring  to gearbox [Nm/rad]  ca. 1Hz: 10.0 2Hz: 36 3Hz:86
% %% NL 1
% parameter.c_1 = 10;
% parameter.c_2 = 2.3065;
% %% NL 2
% % parameter.c_1 = 10;
% % parameter.c_2 = 0.07;
parameter.d_s  = 0.096;     % damping ring  to gearbox [NM/rad s]

% parameter.cw = .3;   % draq coeff [-]

%% Trägheitsterme
%Massenterme
parameter.m_p  = 1*4;     % mass of Planetary wheels (all wheels) [kg]
parameter.m_car = 1300;

%Trägheitsmomente
parameter.J_1  = .1;       % Carrier Intertia [kg m^2]
parameter.J_2  = 0.03;     % Sun wheel Intertia [kg m^2]
parameter.J_3  = 0.1;     % Ring wheel Intertia [kg m^2]
parameter.J_p  = 0.01*4;   % Planetary wheel intertia (all wheels) [kg m^2]
parameter.J_4  = parameter.m_car*parameter.r_Wheel^2 / parameter.g_TT^2;   % Car inertia

%komplexe Trägheitsterme
parameter.M_1 = parameter.J_1 + parameter.J_2 * 4 * parameter.r_1^2/parameter.r_2^2 + parameter.J_p * parameter.r_1^2/parameter.r_p^2 + parameter.m_p * parameter.r_1^2;
parameter.M_2 = parameter.J_2 * 2 * parameter.r_1 * parameter.r_3 / parameter.r_2^2 + parameter.J_p * parameter.r_1 * parameter.r_3 / parameter.r_p^2;
parameter.M_4 = parameter.J_2 * parameter.r_3^2 / parameter.r_2^2 + parameter.J_3 + parameter.J_p * parameter.r_3^2 / parameter.r_p^2;
parameter.M_3 = parameter.J_2 * ( (2 * parameter.r_1 * parameter.r_3 ) / parameter.r_2 ^ 2 ) + parameter.J_p*((parameter.r_1 * parameter.r_3) / parameter.r_p^2);
parameter.M_5 = parameter.M_4 - parameter.M_2*parameter.M_3/parameter.M_1;