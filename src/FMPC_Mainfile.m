% clc; clear; close all;

%addpath(genpath('Linearisierung'))
%addpath(genpath('MPC_Setup'))
%addpath(genpath('Simulation_MPC'))

addpath(genpath(fullfile('..', 'linearization')));
addpath(genpath(fullfile('..', 'mpc_setup')));
addpath(genpath(fullfile('..', 'models')));

%% Zeit
dt = 0.002;
T = 4; 
t = 0:dt:T-dt; 
t = t';

%% Systemdimensionierung (Symbolic Toolbox)
nu = 3;  % Stellgrößen
nxm = 4;  % Zustände
ny = 2;  % Regelgrößen 
nz = 1;  % Störungen

%% Parameter 
parameter=init_parameter_nl();
k_range = [50, 220];

parameter.c_1 = 0.2;
parameter.c_2 = 50;
omega4_range = sqrt((k_range-parameter.c_2)/parameter.c_1); 

c_1 = parameter.c_1;
c_2 = parameter.c_2;

%% Bestimmung des lokal linearen Modellnetzwerkes mittels nugap-Metrik --> 3x Modelle
%     stueckelung_gap = 5;       %% Initialschätzung der Modelle
%     gap_schwell_min = 0.5;  %% bei zu feiner anfänglicher Partionierung, werden die gap-Werte nach oben hin bereinigt
%     gap_schwell_max =  0.5; %% in jedem Fall wird ein idealer gap-Wert angestrebt, um den die gaps konvergieren
%     
%     [Am_lin_global,Am_lin_global_val,Bm_lin,Cm_lin,Em_lin,nugap_final_val,AP_LLM] = Linearisierung(omega4_range,parameter,nxm,nu,ny,nz,stueckelung_gap,gap_schwell_min,gap_schwell_max);
%     fuzzy_menge = nugap_final_val(:,4:5); 

%% Bestimmung des lokal linearen Modellnetzwerkes durch äquidistante Punkte 
    stueckelung_aequi = 4;
    delta_rand = 0.05;

    Am_lin_global_val = zeros(nxm*stueckelung_aequi,nxm);

    AP_LLM = linspace(max(omega4_range)*delta_rand,max(omega4_range)*(1-delta_rand),stueckelung_aequi);
    
    AP_LLM = AP_LLM(2:4); %3x Modelle

    for i = 1:length(AP_LLM)
        [sys_lin,E,A0] = model_lin(nu,nxm,ny,nz,parameter,AP_LLM(i));
        Am_lin_global_val((i-1)*nxm+1:i*nxm,:) = sys_lin.A;
        Bm_lin = sys_lin.B; %bleibt konstant
        Cm_lin = sys_lin.C; %bleibt konstant
        Dm_lin = sys_lin.D; %bleibt konstant
        Em_lin = E; %bleibt konstant
    end

   fuzzy_menge = zeros(length(AP_LLM)-1,2);
   
   for j = 1:length(AP_LLM)-1
       fuzzy_menge(j,2) = AP_LLM(j+1);
       fuzzy_menge(j,1) = AP_LLM(j);
   end


%% Fuzzy-Zugehörigkeitsfunktionen
b = 2; %für b = 0 ... Dreieck; für b > 0 ... Trapez
%% Fuehrungssignale -------------------------------------------------------
n = 1;
%% Ausgang 1 ---------------------------
    %% Sprung
    t_y1_sprung = 0;
    y1_end = 2000/60;
    
    unistep = t >= t_y1_sprung;
    y1_fuehrung = unistep*y1_end;


%% Ausgang 2 ----------------------------------
    %% Sprung auf AP-----------------------------------------
%     t_y2_sprung = 1;
%     y2_sprung =  AP_LLM(n);
% 
%     unistep = t >= t_y2_sprung;
%     y2_fuehrung = unistep*y2_sprung;


    %% Sprung um AP -----------------------------------
%     t_y2_sprung1 = 2;
%     t_y2_sprung2 = 4;
%     y2_sprung =  AP_LLM(n);
% %     y2_sprung =  (AP_LLM(n)+0)/2;
% 
%     AP_umgebung = 0.05;
% 
%     y2_fuehrung = t*0 + (y2_sprung*(1-AP_umgebung));  
%     y2_fuehrung(t > t_y2_sprung1) = (y2_sprung*(2*AP_umgebung)) + (y2_sprung*(1-AP_umgebung));
% %     y2_fuehrung(t > t_y2_sprung2) = -(y2_sprung*(2*AP_umgebung)) + (y2_sprung*(1+AP_umgebung));
% 
%     y2_fuehrung_lin = t*0; 
%     y2_fuehrung_lin(t > t_y2_sprung1) = (y2_sprung*(2*AP_umgebung));
%     y2_fuehrung_lin(t > t_y2_sprung2) = 0;

%% Sprung auf AP-----------------------------------------
%     t_y2_sprung = 1;
%     y2_sprung =  (AP_LLM(n)+AP_LLM(n+1))/2; %Bereiche zw den APs
% %     y2_sprung =  (AP_LLM(n)+0)/2; %Bereich vor AP1
%     
%     unistep = t >= t_y2_sprung;
%     y2_fuehrung = unistep*y2_sprung;


    %% Rampe ------------------------------------------
%     t_y2_rampe1 = 1;
%     t_y2_rampe2 = 4;
%     
%     y2_rampe1 = AP_LLM(1);
%     y2_rampe2 = AP_LLM(n);
%     
%     steigung = ((y2_rampe2-y2_rampe1)/(t_y2_rampe2-t_y2_rampe1));
%     y2_fuehrung = t*0;  
%     
%     y2_fuehrung(t < t_y2_rampe1) = y2_rampe1;
%     y2_fuehrung(t_y2_rampe1/dt+1:t_y2_rampe2/dt+1) = t(t_y2_rampe1/dt+1:t_y2_rampe2/dt+1)*steigung+(y2_rampe1-t_y2_rampe1*steigung);
%     y2_fuehrung(t > t_y2_rampe2) = y2_rampe2;


    %% Sinus mit Modulation der Frequenz und Amplitude
%     t_sinus = 0:dt:5.25;
%     t_1 = 0:dt:1;
% 
% %     y2_sprung =  AP_LLM(n);
%     y2_sprung =  2*(AP_LLM(n)+AP_LLM(n+1))/3;
% 
%     unistep = t>=t_sinus(1);
%     ramp = t_sinus.*unistep*y2_sprung/t_sinus(end)/2;
%     
%     y2_sinus = ramp(1,:).*sin(2*pi*(2*t_sinus-cos(t_sinus)))+y2_sprung/2;
%     y2_fuehrung = [zeros(length(t_1),1)+y2_sprung/2; y2_sinus'; zeros(length(t)-length(t_sinus)-length(t_1),1)+y2_sprung/2];

%% 1. Sprungszenario - Herumspringen um AP
%     t_y2_sprung1 = 1;
%     t_y2_sprung2 = 2;
%     t_y2_sprung3 = 3;
%     y2_sprung =  AP_LLM(n);
% %     y2_sprung =  (AP_LLM(n)+0)/2;
% 
%     AP_umgebung = 0.05;
% 
%     y2_fuehrung = t*0 + y2_sprung;  
%     y2_fuehrung(t > t_y2_sprung1) = (y2_sprung*(1+AP_umgebung));
%     y2_fuehrung(t > t_y2_sprung2) = -(y2_sprung*(2*AP_umgebung)) + (y2_sprung*(1+AP_umgebung));
%     y2_fuehrung(t > t_y2_sprung3) = y2_sprung; 

    %% 2. Sprungszenario - Sprung zw. APs
%     t_y2_sprung1 = 1;
%     y2_sprung =  (AP_LLM(n)+AP_LLM(n+1))/2; %Bereiche zw den APs
% %     y2_sprung =  (AP_LLM(n)+0)/2; %Bereich vor AP1
%     
%     y2_fuehrung = t*0 + AP_LLM(n); 
%     y2_fuehrung(t > t_y2_sprung1) = y2_sprung;

    %% 3. Sprungszenario - Sprung zw. APs/nicht-stationär
     t_y2_sprung1 = 2;
     t_y2_sprung2 = 2.5;
    
    y2_fuehrung = t*0 + AP_LLM(1); 
    y2_fuehrung(t > t_y2_sprung1) = AP_LLM(3);
    y2_fuehrung(t > t_y2_sprung2) = AP_LLM(1);

%% Störungen
t1z = 2.5;
z_end = 0;
% z_end = 20000/60;

unistep = t >= t1z;
ztrack = unistep*z_end;

%% Modellerhebung
l = length(Am_lin_global_val)/nxm;

Am_c_global = Am_lin_global_val(1:nxm*l,:);
Bm = Bm_lin;
Cm = Cm_lin;
Em = Em_lin; 

Am_d_global = zeros(nxm*l,nxm);
Bm_d_global = zeros(nxm*l,nu);

for i = 1:l
    Am = Am_c_global((i-1)*nxm+1:i*nxm,:);
    sys_d = c2d(ss(Am,[Bm Em],Cm,0),dt);

    Am_d_global((i-1)*nxm+1:i*nxm,:) = sys_d.A;
end

Bm = sys_d.B(:,1:3);
Em = sys_d.B(:,4); 

%% MPC-Setup --------------------------------------------------------------
%% MPC-Parameter
Np = 50;  % prediction horizon
Nc = 15;  % controller horizon

%% Gewichtung
Q_global = zeros(l,ny);
R_global = zeros(l,nu);

Qy_global = zeros(Np*ny*l,Np*ny);
Ru_global = zeros(Nc*nu*l,Nc*nu);

%% Globale Gewichtung
%Ohne Beschränkungen
% Q  = [1e0 1e1]*82500;
% R  = [1e0 1e0 1e0]*1;

% Q  = [1e0 1e1]*10000;
% R  = [1e0 1e0 1e0]*1;
%Beschränkungen
Q  = [1e0 1e1]*82500;
R  = [1e0 1e0 1e0]*1000;
% Q  = [1e0 1e1]*82500;
% R  = [1e0 1e0 1e0]*70000;

Qy = diag(repmat(Q,1,Np));
Ru = diag(repmat(R,1,Nc));

Q_global(1,:) = Q;
R_global(1,:) = R;

Q_global(2,:) = Q;
R_global(2,:) = R;

Q_global(3,:) = Q;
R_global(3,:) = R;

if l == 4
    Q_global(4,:) = Q;
    R_global(4,:) = R;
end

%% Lokale Gewichtung
% Q_global(1,:) = [1e0 1e1]*1000;
% R_global(1,:) = [1e0 1e0 1e0]*0.01;
% 
% Q_global(2,:) = [1e0 1e1]*10000;
% R_global(2,:) = [1e0 1e0 1e0]*0.01;
% 
% Q_global(3,:) = [1e0 1e1]*100;
% R_global(3,:) = [1e0 1e0 1e0]*0.1;
% 
% if l == 4
%     Q_global(4,:) = [1e0 1e1]*1000;
%     R_global(4,:) = [1e0 1e0 1e0]*0.01;
% end

%% Globale Variable
for k = 1:l
    Qy_global(1+(k-1)*Np*ny:k*Np*ny,:) = diag(repmat(Q_global(k,:),1,Np));
    Ru_global(1+(k-1)*Nc*nu:k*Nc*nu,:) = diag(repmat(R_global(k,:),1,Nc));
end


%% Beschränkungen festlegen -------------------- --> alle MPCs gleich

%% inkrementelle Beschränkung
du_lim = [
    NaN, NaN      % min, max delta_u1
    NaN, NaN      % min, max delta_u2
    NaN, NaN      % min, max delta_u3
    ];    

%% Beschränkung der Amplitude
u_lim = [
%     -790, 820         % min, max u1
%     -790, 820         % min, max u2
%     -790, 820         % min, max u3
    -850, 850         % min, max u1
    -850, 850         % min, max u2
    -850, 850         % min, max u3
    ];     

%% Beschränkung am Ausgang
y_lim = [
    1000/60, 4500/60       %min, max y1
    NaN, NaN       %min, max y2
    ];     


%% Deaktivierung der Beschränkungen
du_lim = du_lim*NaN;
% u_lim  = u_lim*NaN;
y_lim  = y_lim*NaN;

dUmin = repmat(du_lim(:,1),Nc,1); 
dUmax = repmat(du_lim(:,2),Nc,1);

Umin  = repmat(u_lim(:,1),Nc,1);
Umax  = repmat(u_lim(:,2),Nc,1);

Ymin  = repmat(y_lim(:,1),Np,1); 
Ymax  = repmat(y_lim(:,2),Np,1);

%% Modellerweiterung
[nxm,nu] = size(Bm);
ny = size(Cm,1);  
nx = ny + nxm;

A_global = zeros((nxm+ny)*l,nxm+ny);
B_global = zeros((nxm+ny)*l,nu);

for i = 1:l
    Am = Am_d_global((i-1)*nxm+1:i*nxm,:);
    [A,B,C,E] = mpc_modellerweiterung(Am,Bm,Cm,Em);

    A_global((nxm+ny)*(i-1)+1:i*(nxm+ny),:) = A;
end
D = zeros(nxm,nu);
C_hilf = eye(nxm);

%% MPC-Matrizen
F_global = zeros(Np*ny*l,nx);
PhiU_global = zeros(Np*ny*l,Nc*nu);
PhiZ_global = zeros(Np*ny*l,Np*nz);
H_global = zeros(Nc*nu*l,Nc*nu);

for ii = 1:l
    A = A_global((nxm+ny)*(ii-1)+1:ii*(nxm+ny),:);
    Qy = Qy_global(1+(ii-1)*Np*ny:ii*Np*ny,:);
    Ru = Ru_global(1+(ii-1)*Nc*nu:ii*Nc*nu,:);

    [F,PhiU,PhiZ,H] = mpc_matrizen(A,B,C,E,Np,Nc,Qy,Ru);

    F_global( (ii-1)*Np*ny+1 : ii*Np*ny, : ) = F;
    PhiU_global( (ii-1)*Np*ny+1 : ii*Np*ny , :) = PhiU;
    PhiZ_global( (ii-1)*Np*ny+1 : ii*Np*ny, : ) = PhiZ;
    H_global( (ii-1)*Nc*nu+1 : ii*Nc*nu, : ) = H;
end

%% Initialisierung
N = numel(t) + Np + 1;

x1 = zeros(nx, N);
u1 = zeros(nu, N);
dU1 = zeros(nu*Nc,1);
du1 = zeros(nu, N);
y1 = zeros(ny, N);

x2 = zeros(nx, N);
u2 = zeros(nu, N);
dU2 = zeros(nu*Nc,1);
du2 = zeros(nu, N);
y2 = zeros(ny, N);

x3 = zeros(nx, N);
u3 = zeros(nu, N);
dU3 = zeros(nu*Nc,1);
du3 = zeros(nu, N);
y3 = zeros(ny, N);

x4 = zeros(nx, N);
u4 = zeros(nu, N);
dU4 = zeros(nu*Nc,1);
du4 = zeros(nu, N);
y4 = zeros(ny, N);

ztrack = [ztrack;  ztrack(end)*ones(2*Np,1)];
dztrack = [0; diff(ztrack)];
dZ  = zeros(nz*Np,1);

%% Aufbereitung Führungsgröße
y1_fuehrung =  [y1_fuehrung; y1_fuehrung(end)*ones(2*Np,1)];
y2_fuehrung = [y2_fuehrung; y2_fuehrung(end)*ones(2*Np,1)];
Y_fuehrung = zeros(ny*Np,1);

Y_fuehrung_SIM = zeros(N-1,ny*Np+1);
Y_fuehrung_SIM(:,1) = 1:(N-1);
Y_fuehrung_SIM(:,1) = Y_fuehrung_SIM(:,1)*dt;

ztrack_SIM = zeros(N-1,nz);
ztrack_SIM(:,1) = 1:(N-1);
ztrack_SIM(:,1) = ztrack_SIM(:,1)*dt;

dztrack_SIM = zeros(N-1,nz*Np+1);
dztrack_SIM(:,1) = 1:(N-1);
dztrack_SIM(:,1) = dztrack_SIM(:,1)*dt;

for i = 1:N-1
    Y_fuehrung_SIM(i,2:2:end) = y1_fuehrung(i:i+Np-1);
    Y_fuehrung_SIM(i,3:2:end) = y2_fuehrung(i:i+Np-1);
    dztrack_SIM(i,2:end) = dztrack(i:i+Np-1);
    ztrack_SIM(i,2) = ztrack(i);
end


%% Simulation -----------------------------------------------
A1 = A_global(1:(nxm+ny),:);
A2 = A_global((nxm+ny)+1:2*(nxm+ny),:);
A3 = A_global(2*(nxm+ny)+1:3*(nxm+ny),:);

if l == 4
    A4 = A_global(3*(nxm+ny)+1:4*(nxm+ny),:);
end

Qy1 = Qy_global(1:1*Np*ny,:);
Ru1 = Ru_global(1:1*Nc*nu,:);

Qy2 = Qy_global(1+1*Np*ny:2*Np*ny,:);
Ru2 = Ru_global(1+1*Nc*nu:2*Nc*nu,:);

Qy3 = Qy_global(1+2*Np*ny:3*Np*ny,:);
Ru3 = Ru_global(1+2*Nc*nu:3*Nc*nu,:);

if l == 4
    Qy4 = Qy_global(1+3*Np*ny:4*Np*ny,:);
    Ru4 = Ru_global(1+3*Nc*nu:4*Nc*nu,:);
end

F1 = F_global(1:Np*ny,:);
PhiU1 = PhiU_global(1:Np*ny,:);
PhiZ1 = PhiZ_global(1:Np*ny,:);
H1 = H_global(1:Nc*nu,:);

F2 = F_global(Np*ny+1:Np*ny*2,:);
PhiU2 = PhiU_global(Np*ny+1:2*Np*ny,:);
PhiZ2 = PhiZ_global(Np*ny+1:2*Np*ny,:);
H2 = H_global(Nc*nu+1:Nc*nu*2,:);

F3 = F_global(2*Np*ny+1:Np*ny*3,:);
PhiU3 = PhiU_global(2*Np*ny+1:Np*ny*3,:);
PhiZ3 = PhiZ_global(2*Np*ny+1:Np*ny*3,:);
H3 = H_global(2*Nc*nu+1:Nc*nu*3,:);

if l == 4
    F4 = F_global(3*Np*ny+1:4*Np*ny,:);
    PhiU4 = PhiU_global(3*Np*ny+1:4*Np*ny,:);
    PhiZ4 = PhiZ_global(3*Np*ny+1:4*Np*ny,:);
    H4 = H_global(3*Nc*nu+1:4*Nc*nu,:);
end

%% Beschränkungsmatrizen ---------------------------
C1 = repmat(eye(nu),Nc,1);     
C2 = tril(repmat(eye(nu),Nc));
M_du = [-eye(nu*Nc); eye(nu*Nc)];
M_u  = [-C2; C2]; 
gamma_du = [-dUmin; dUmax];

%% SIMULATION -----------------------------------------------------------------
%% Solver Options %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
my_opt = optimoptions('quadprog');
my_opt.Algorithm = 'interior-point-convex';
% my_opt.Algorithm = 'active-set';
my_opt.Display = 'off';
% my_opt.MaxIterations = 1000;
% my_opt.OptimalityTolerance = 1e-5;
% my_opt.ConstraintTolerance = 1e-5;


%% FMPC
fmpc_sim_out = sim('fmpc_simulation_3Modelle.slx');

%fmpc_sim_out = sim('fmpc_simulation_3Modelle_mit_Beschraenkung.slx');
% fmpc_sim_out_mit_Beschraenkung = sim('fmpc_simulation_3Modelle_mit_Beschraenkung.slx');

% fmpc_sim_out_ohne_z = sim('fmpc_simulation_3modelle_ohne_z.slx');
% 
% mpc1_sim_out = sim('mpc1_simulation_mitbeschraenkung.slx');

% mpc1_sim_out = sim('mpc1_simulation.slx');
% mpc2_sim_out = sim('mpc2_simulation.slx');
% mpc3_sim_out = sim('mpc3_simulation.slx');
% mpc4_sim_out = sim('mpc4_simulation.slx');

                %% Validierung Nc
                % tic
                % mpc1_sim_out = sim('mpc1_simulation.slx');
                % toc
                
                % tic
                % mpc1_sim_out_Nc5 = sim('mpc1_simulation.slx');
                % toc
                % 
                % tic
                % mpc1_sim_out_Nc10 = sim('mpc1_simulation.slx');
                % toc
                % 
                % tic
                % mpc1_sim_out_Nc25 = sim('mpc1_simulation.slx');
                % toc
                % tic
                % mpc1_sim_out_Nc50 = sim('mpc1_simulation.slx');
                % toc

%% MPC - auf linearer Strecke
k = 1;

x1(:,k+1)= A1*x1(:,k) + B*du1(:,k); 
y1(:,k) = C*x1(:,k);

x2(:,k+1)= A2*x2(:,k) + B*du2(:,k); 
y2(:,k) = C*x2(:,k);

x3(:,k+1)= A3*x3(:,k) + B*du3(:,k); 
y3(:,k) = C*x3(:,k);

if l == 4
    x4(:,k+1)= A4*x4(:,k) + B*du4(:,k); 
    y4(:,k) = C*x4(:,k);
end

for k = 2:N-1
    Y_fuehrung(1:2:end) = y1_fuehrung(k:k+Np-1);
    Y_fuehrung(2:2:end) = y2_fuehrung(k:k+Np-1);

    dZ(1:end) = dztrack(k:k+Np-1);

    %% MPC1 --------------------------------------------------
    M_y  = [-PhiU1; PhiU1];   
    M = [M_u;M_du;M_y];
    
    gamma_du = [-dUmin; dUmax];
    gamma_u = [ -Umin + C1*u1(:,k-1); Umax - C1*u1(:,k-1)];
    gamma_y = [ -Ymin + F1*x1(:,k) + PhiZ1*dZ; Ymax - F1*x1(:,k) - PhiZ1*dZ]; 
    gamma = [gamma_u;gamma_du;gamma_y];
    
    M(isnan(gamma),:) = []; 
    gamma(isnan(gamma)) = [];

    f1 = -PhiU1' * Qy1 * ( Y_fuehrung - F1 * x1(:,k) - PhiZ1*dZ);
    dU1 = -H1\f1;
%     dU1 = quadprog(H1,f1,M,gamma,[],[],[],[],dU1,my_opt);
    du1(:,k) = dU1(1:nu,1); 

    x1(:,k+1)= A1*x1(:,k) + B*du1(:,k) + E*dZ(1:nz,1); 
    y1(:,k) = C*x1(:,k);

    u1(:,k) = u1(:,k-1) + du1(:,k);


    %% MPC2 --------------------------------------------------
    M_y  = [-PhiU2; PhiU2];   
    M = [M_u;M_du;M_y];
    
    gamma_du = [-dUmin; dUmax];
    gamma_u = [ -Umin + C1*u2(:,k-1); Umax - C1*u2(:,k-1)];
    gamma_y = [ -Ymin + F2*x2(:,k) + PhiZ2*dZ; Ymax - F2*x2(:,k) - PhiZ2*dZ]; 
    gamma = [gamma_u;gamma_du;gamma_y];
    
    M(isnan(gamma),:) = []; 
    gamma(isnan(gamma)) = [];

    f2 = -PhiU2' * Qy2 * ( Y_fuehrung - F2 * x2(:,k) - PhiZ2*dZ);
    dU2 = -H2\f2;
%     dU2 = quadprog(H2,f2,M,gamma,[],[],[],[],dU2,my_opt);
    du2(:,k) = dU2(1:nu,1);

    x2(:,k+1)= A2*x2(:,k) + B*du2(:,k) + E*dZ(1:nz,1); 
    y2(:,k) = C*x2(:,k);

    u2(:,k) = u2(:,k-1) + du2(:,k);


%     %% MPC3 --------------------------------------------------
    M_y  = [-PhiU3; PhiU3];   
    M = [M_u;M_du;M_y];
    
    gamma_du = [-dUmin; dUmax];
    gamma_u = [ -Umin + C1*u3(:,k-1); Umax - C1*u3(:,k-1)];
    gamma_y = [ -Ymin + F3*x3(:,k) + PhiZ3*dZ; Ymax - F3*x3(:,k) - PhiZ3*dZ]; 
    gamma = [gamma_u;gamma_du;gamma_y];
    
    M(isnan(gamma),:) = []; 
    gamma(isnan(gamma)) = [];

    f3 = -PhiU3' * Qy3 * ( Y_fuehrung - F3 * x3(:,k) - PhiZ3*dZ);
    dU3 = -H3\f3;
%     dU3 = quadprog(H3,f3,M,gamma,[],[],[],[],dU3,my_opt);
    du3(:,k) = dU3(1:nu,1);

    x3(:,k+1)= A3*x3(:,k) + B*du3(:,k) + E*dZ(1:nz,1); 
    y3(:,k) = C*x3(:,k);

    u3(:,k) = u3(:,k-1) + du3(:,k);
% 
% 
%     %% MPC4 --------------------------------------------------
%     M_y  = [-PhiU4; PhiU4];   
%     M = [M_u;M_du;M_y];
%     
%     gamma_du = [-dUmin; dUmax];
%     gamma_u = [ -Umin + C1*u4(:,k-1); Umax - C1*u4(:,k-1)];
%     gamma_y = [ -Ymin + F4*x4(:,k) + PhiZ4*dZ; Ymax - F4*x4(:,k) - PhiZ4*dZ]; 
%     gamma = [gamma_u;gamma_du;gamma_y];
%     
%     M(isnan(gamma),:) = []; 
%     gamma(isnan(gamma)) = [];
% 
%     if l == 4
%         f4 = -PhiU4' * Qy4 * ( Y_fuehrung - F4 * x4(:,k) - PhiZ4*dZ);
% %         dU4 = -H4\f4;
%         dU4 = quadprog(H4,f4,M,gamma,[],[],[],[],dU4,my_opt);
%         du4(:,k) = dU4(1:nu,1);
%     
%         x4(:,k+1)= A4*x4(:,k) + B*du4(:,k) + E*dZ(1:nz,1); 
%         y4(:,k) = C*x4(:,k);
% 
%         u4(:,k) = u4(:,k-1) + du4(:,k);
%     end
end


%% Plot
% figure; tiledlayout(4,1);
figure; tiledlayout(3,1); 
% figure; tiledlayout(2,1);
nexttile; hold on; box on; 
line11 = plot(t,y1_fuehrung(1:numel(t)),'k','DisplayName','Set Point','LineWidth',2);
% line12 = plot(t,y1(1,1:numel(t)),'y-','DisplayName','MPC: LLM 1 on Linear Plant','LineWidth',1);
% line13 = plot(t,y2(1,1:numel(t)),'y-','DisplayName','MPC: LLM 2 on Linear Plant','LineWidth',1);
% line14 = plot(t,y3(1,1:numel(t)),'y-','DisplayName','MPC: LLM 3 on Linear Plant','LineWidth',1);
% % plot(t,y4(1,1:numel(t)),'y-','DisplayName','MPC: LLM 4 on Linear Plant','LineWidth',1);
%             plot(t,mpc1_sim_out.y1_llm1(1:numel(t)),'-','Color','#77AC30','DisplayName','MPC (Nc = 15): LLM 1 on Non-Linear Plant','LineWidth',1);
%             plot(t,mpc1_sim_out_Nc5.y1_llm1(1:numel(t)),'-','Color','#7E2F8E','DisplayName','MPC (Nc = 5): LLM 1 on Non-Linear Plant','LineWidth',1);
%             plot(t,mpc1_sim_out_Nc10.y1_llm1(1:numel(t)),'-','Color','#4DBEEE','DisplayName','MPC (Nc = 10): LLM 1 on Non-Linear Plant','LineWidth',1);
%             plot(t,mpc1_sim_out_Nc25.y1_llm1(1:numel(t)),'-','Color','#EDB120','DisplayName','MPC (Nc = 25): LLM 1 on Non-Linear Plant','LineWidth',1);
%             plot(t,mpc1_sim_out_Nc50.y1_llm1(1:numel(t)),'-','Color','#EDB120','DisplayName','MPC (Nc = 50): LLM 1 on Non-Linear Plant','LineWidth',1);
%     line17 = plot(t,mpc1_sim_out.y1_llm1(1:numel(t)),'-','Color','#77AC30','DisplayName','MPC: LLM 1 on Non-Linear Plant','LineWidth',1);
%     line18 = plot(t,mpc2_sim_out.y1_llm2(1:numel(t)),'-','Color','#77AC30','DisplayName','MPC: LLM 2 on Non-Linear Plant','LineWidth',1);
%     line19 =plot(t,mpc3_sim_out.y1_llm3(1:numel(t)),'-','Color','#77AC30','DisplayName','MPC: LLM 3 on Non-Linear Plant','LineWidth',1);
% %     plot(t,mpc4_sim_out.y1_llm4(1:numel(t)),'-','Color','#EDB120','DisplayName','MPC: LLM 4 on Non-Linear Plant','LineWidth',1);
    line110 = plot(t,fmpc_sim_out.y1(1:numel(t)),'r','DisplayName','FMPC','LineWidth',2);
%            line111 = plot(t,fmpc_sim_out_ohne_z.y1(1:numel(t)),'g:','DisplayName','FMPC without Disturbance\newlineSuppression','LineWidth',2);
p = yline(1000/60,'--','DisplayName','Lower Limit of Efficient Operating Regime'); p.Annotation.LegendInformation.IconDisplayStyle = 'off';
p = yline(4500/60,'--','DisplayName','Upper Limit of Efficient Operating Regime'); p.Annotation.LegendInformation.IconDisplayStyle = 'off';
% p = plot(t,y_lim(1,1)+t*0,'k--','DisplayName','Beschränkung','LineWidth',0.5); p.Annotation.LegendInformation.IconDisplayStyle = 'off';
% p = plot(t,y_lim(1,2)+t*0,'k--','DisplayName','Beschränkung','LineWidth',0.5); p.Annotation.LegendInformation.IconDisplayStyle = 'off';
title('System Output 1: Internal Combustion Engine - Angular Velocity \omega_1','FontSize',12)
% legend('Location','best');
set(gca,'FontSize',12)

nexttile; hold on; box on; 
line21 = plot(t,y2_fuehrung(1:numel(t)),'k','DisplayName','Set Point','LineWidth',2);
% line22 = plot(t,y1(2,1:numel(t)),'y-','DisplayName','MPC: LLM 1 on Linear Plant','LineWidth',1);
% line23 = plot(t,y2(2,1:numel(t)),'y-','DisplayName','MPC: LLM 2 on Linear Plant','LineWidth',1);
% line24 =plot(t,y3(2,1:numel(t)),'y-','DisplayName','MPC: LLM 3 on Linear Plant','LineWidth',1);
% plot(t,y4(2,1:numel(t)),'y-','DisplayName','MPC: LLM 4 on Linear Plant','LineWidth',1);
%             plot(t,mpc1_sim_out.y2_llm1(1:numel(t)),'-','Color','#77AC30','DisplayName','MPC (Nc = 15): LLM 1 on Non-Linear Plant','LineWidth',1);
%             plot(t,mpc1_sim_out_Nc5.y2_llm1(1:numel(t)),'-','Color','#7E2F8E','DisplayName','MPC (Nc = 5): LLM 1 on Non-Linear Plant','LineWidth',1);
%             plot(t,mpc1_sim_out_Nc10.y2_llm1(1:numel(t)),'-','Color','#4DBEEE','DisplayName','MPC (Nc = 10): LLM 1 on Non-Linear Plant','LineWidth',1);
%             plot(t,mpc1_sim_out_Nc25.y2_llm1(1:numel(t)),'-','Color','#EDB120','DisplayName','MPC (Nc = 25): LLM 1 on Non-Linear Plant','LineWidth',1);
%             plot(t,mpc1_sim_out_Nc50.y2_llm1(1:numel(t)),'-','Color','#EDB120','DisplayName','MPC (Nc = 50): LLM 1 on Non-Linear Plant','LineWidth',1);
%     plot(t,mpc1_sim_out.y2_llm1(1:numel(t)),'-','Color','#77AC30','DisplayName','MPC: LLM 1 on Non-Linear Plant','LineWidth',1);
%     plot(t,mpc2_sim_out.y2_llm2(1:numel(t)),'-','Color','#77AC30','DisplayName','MPC: LLM 2 on Non-Linear Plant','LineWidth',1);
%     plot(t,mpc3_sim_out.y2_llm3(1:numel(t)),'-','Color','#77AC30','DisplayName','MPC: LLM 3 on Non-Linear Plant','LineWidth',1);
% %     plot(t,mpc4_sim_out.y2_llm4(1:numel(t)),'-','Color','#77AC30','DisplayName','MPC: LLM 4 on Non-Linear Plant','LineWidth',1);
    plot(t,fmpc_sim_out.y2(1:numel(t)),'r','DisplayName','FMPC','LineWidth',2);
%             plot(t,fmpc_sim_out_ohne_z.y2(1:numel(t)),'g:','DisplayName','FMPC without Disturbance\newlineSuppression','LineWidth',2);
p = yline(AP_LLM(n),'-.','DisplayName','Operating Point'); p.Annotation.LegendInformation.IconDisplayStyle = 'off';
p = yline(AP_LLM(n+1),'-.','DisplayName','Operating Point'); p.Annotation.LegendInformation.IconDisplayStyle = 'off';
p = yline(AP_LLM(n+2),'-.','DisplayName','Operating Point'); p.Annotation.LegendInformation.IconDisplayStyle = 'off';
% p = plot(t,y_lim(2,1)+t*0,'k--','DisplayName','Beschränkung','LineWidth',0.5); p.Annotation.LegendInformation.IconDisplayStyle = 'off';
% p = plot(t,y_lim(2,2)+t*0,'k--','DisplayName','Beschränkung','LineWidth',0.5); p.Annotation.LegendInformation.IconDisplayStyle = 'off';
title('System Output 2: (reduced) Rotational Mass - Angular Velocity \omega_4','FontSize',12)
% legend('Location','best');
set(gca,'FontSize',12)

nexttile; hold on; box on; 
line31 = plot(t,fmpc_sim_out.phi_fuzzy1(1:numel(t),1)','g-','DisplayName','\Phi_1','LineWidth',1);
line32 = plot(t,fmpc_sim_out.phi_fuzzy2(1:numel(t),1)','b-','DisplayName','\Phi_2','LineWidth',1);
line33 = plot(t,fmpc_sim_out.phi_fuzzy3(1:numel(t),1)','c-','DisplayName','\Phi_3','LineWidth',1);
% % % plot(t,fmpc_sim_out.phi_fuzzy4(1:numel(t),1)','y-','DisplayName','\Phi_4','LineWidth',1);
%         line34 = plot(t,fmpc_sim_out_ohne_z.phi_fuzzy1(1:numel(t),1)',':','Color','#77AC30','DisplayName','\Phi_1 (without Disturbance\newlineSuppression)','LineWidth',1);
%         line35 = plot(t,fmpc_sim_out_ohne_z.phi_fuzzy2(1:numel(t),1)',':','Color','#0072BD','DisplayName','\Phi_2 (without Disturbance\newlineSuppression)','LineWidth',1);
%         line36 = plot(t,fmpc_sim_out_ohne_z.phi_fuzzy3(1:numel(t),1)',':','Color','#4DBEEE','DisplayName','\Phi_3 (without Disturbance\newlineSuppression)','LineWidth',1);
title('Membership Functions ','FontSize',10)
xlabel('Time (s)')
% legend('Location','best');
set(gca,'FontSize',12)

% nexttile; hold on; box on; 
% plot(t,ztrack(1:numel(t)),'r','DisplayName','Disturbance','LineWidth',2)
% title('System Disturbance: (reduced) Rotational Mass - Torque T_{Load}')
% linkaxes(findall(gcf,'type','axes'),'x')
% legend('Location','best');

linkaxes(findall(gcf,'type','axes'),'x')
% lgd = legend(nexttile(2), [line11,line12,line17,line18],'NumColumns',2);
% lgd.Layout.Tile = 'south';
% lgd = legend(nexttile(2), [line11,line12,line17,line110,line31,line32,line33]);
% lgd = legend(nexttile(2), [line11,line13,line18,line110,line31,line32,line33]);
% lgd = legend(nexttile(2), [line11,line14,line19,line110,line31,line32,line33]);
lgd = legend(nexttile(2), [line11,line110,line31,line32,line33]);
% lgd = legend(nexttile(2), [line11,line110,line111,line31,line32,line33,line34,line35,line36]);
lgd.Layout.Tile = 'east';
lgd.FontSize = 12;
set(gca,'FontSize',12)

% figure; tiledlayout(5,1); 
% figure; tiledlayout(4,1); 
figure; tiledlayout(3,1); 
nexttile; hold on; box on; 
% line11 = plot(t,u1(1,1:numel(t)),'y-','DisplayName','MPC: LLM 1 on Linear Plant','LineWidth',1);
% line12 = plot(t,u2(1,1:numel(t)),'y-','DisplayName','MPC: LLM 2 on Linear Plant','LineWidth',1);
% line13 = plot(t,u3(1,1:numel(t)),'y-','DisplayName','MPC: LLM 3 on Linear Plant','LineWidth',1);
% % % plot(t,u4(1,1:numel(t)),'r-','DisplayName','MPC: LLM 4 on Linear Plant','LineWidth',1);
%             plot(t,mpc1_sim_out.u1_llm1(1:numel(t)),'-','Color','#77AC30','DisplayName','MPC (Nc = 15): LLM 1 on Non-Linear Plant','LineWidth',1);
%             plot(t,mpc1_sim_out_Nc5.u1_llm1(1:numel(t)),'-','Color','#7E2F8E','DisplayName','MPC (Nc = 5): LLM 1 on Non-Linear Plant','LineWidth',1);
%             plot(t,mpc1_sim_out_Nc10.u1_llm1(1:numel(t)),'-','Color','#4DBEEE','DisplayName','MPC (Nc = 10): LLM 1 on Non-Linear Plant','LineWidth',1);
%             plot(t,mpc1_sim_out_Nc25.u1_llm1(1:numel(t)),'-','Color','#EDB120','DisplayName','MPC (Nc = 25): LLM 1 on Non-Linear Plant','LineWidth',1);
%             plot(t,mpc1_sim_out_Nc50.u1_llm1(1:numel(t)),'-','Color','#EDB120','DisplayName','MPC (Nc = 25): LLM 1 on Non-Linear Plant','LineWidth',1);
%     line14 = plot(t,mpc1_sim_out.u1_llm1(1:numel(t)),'-','Color','#77AC30','DisplayName','MPC: LLM 1 on Non-Linear Plant','LineWidth',1);
%     line15 = plot(t,mpc2_sim_out.u1_llm2(1:numel(t)),'-','Color','#77AC30','DisplayName','MPC: LLM 2 on Non-Linear Plant','LineWidth',1);
%     line16 = plot(t,mpc3_sim_out.u1_llm3(1:numel(t)),'-','Color','#77AC30','DisplayName','MPC: LLM 3 on Non-Linear Plant','LineWidth',1);
% %     plot(t,mpc4_sim_out.u1_llm4(1:numel(t)),'-','Color','#EDB120','DisplayName','MPC: LLM 4 on Non-Linear Plant','LineWidth',1);
% line19 = plot(t,fmpc_sim_out.u_llm1_1(1:numel(t)),'g-','DisplayName','FMPC: Linear Local Model 1','LineWidth',1);
% line110 = plot(t,fmpc_sim_out.u_llm2_1(1:numel(t)),'b-','DisplayName','FMPC: Linear Local Model 2','LineWidth',1);
% line111 = plot(t,fmpc_sim_out.u_llm3_1(1:numel(t)),'c-','DisplayName','FMPC: Linear Local Model 3','LineWidth',1);
%     plot(t,fmpc_sim_out_ohne_z.u_llm1_1(1:numel(t)),'--','Color','#77AC30','DisplayName','FMPC without Disturbance Suppression: Linear Local Model 1','LineWidth',1);
%     plot(t,fmpc_sim_out_ohne_z.u_llm2_1(1:numel(t)),'--','Color','#0072BD','DisplayName','FMPC without Disturbance Suppression: Linear Local Model 2','LineWidth',1);
%     plot(t,fmpc_sim_out_ohne_z.u_llm3_1(1:numel(t)),'--','Color','#4DBEEE','DisplayName','FMPC without Disturbance Suppression: Linear Local Model 3','LineWidth',1);
% % plot(t,fmpc_sim_out.u_llm4_1(1:numel(t)),'y--','DisplayName','FMPC: Linear Local Model 4','LineWidth',1);
    line17 = plot(t,fmpc_sim_out.u1(1:numel(t)),'r','DisplayName','FMPC','LineWidth',2);
%         line18 = plot(t,fmpc_sim_out_ohne_z.u1(1:numel(t)),'g:','DisplayName','FMPC without Disturbance\newlineSuppression','LineWidth',2);
p = plot(t,u_lim(1,1)+t*0,'k--','DisplayName','Beschränkung','LineWidth',0.5); p.Annotation.LegendInformation.IconDisplayStyle = 'off';
p = plot(t,u_lim(1,2)+t*0,'k--','DisplayName','Beschränkung','LineWidth',0.5); p.Annotation.LegendInformation.IconDisplayStyle = 'off';
title('System Input 1: Internal Combustion Engine - Torque T_{Eng}','FontSize',12)
% legend('Location','best');
set(gca,'FontSize',12)

nexttile; hold on; box on; 
% line21 = plot(t,u1(2,1:numel(t)),'y-','DisplayName','MPC: LLM 1 on Linear Plant','LineWidth',1);
% line22 = plot(t,u2(2,1:numel(t)),'y-','DisplayName','MPC: LLM 2 on Linear Plant','LineWidth',1);
% plot(t,u3(2,1:numel(t)),'y-','DisplayName','MPC: LLM 3 on Linear Plant','LineWidth',1);
% plot(t,u4(2,1:numel(t)),'r-','DisplayName','MPC: LLM 4 on Linear Plant','LineWidth',1);
%             plot(t,mpc1_sim_out.u2_llm1(1:numel(t)),'-','Color','#77AC30','DisplayName','MPC (Nc = 15): LLM 1 on Non-Linear Plant','LineWidth',1);
%             plot(t,mpc1_sim_out_Nc5.u2_llm1(1:numel(t)),'-','Color','#7E2F8E','DisplayName','MPC (Nc = 5): LLM 1 on Non-Linear Plant','LineWidth',1);
%             plot(t,mpc1_sim_out_Nc10.u2_llm1(1:numel(t)),'-','Color','#4DBEEE','DisplayName','MPC (Nc = 10): LLM 1 on Non-Linear Plant','LineWidth',1);
%             plot(t,mpc1_sim_out_Nc25.u2_llm1(1:numel(t)),'-','Color','#EDB120','DisplayName','MPC (Nc = 25): LLM 1 on Non-Linear Plant','LineWidth',1);
%             plot(t,mpc1_sim_out_Nc50.u2_llm1(1:numel(t)),'-','Color','#EDB120','DisplayName','MPC (Nc = 25): LLM 1 on Non-Linear Plant','LineWidth',1);
%     plot(t,mpc1_sim_out.u2_llm1(1:numel(t)),'-','Color','#77AC30','DisplayName','MPC: LLM 1 on Non-Linear Plant','LineWidth',1);
%     plot(t,mpc2_sim_out.u2_llm2(1:numel(t)),'-','Color','#77AC30','DisplayName','MPC: LLM 2 on Non-Linear Plant','LineWidth',1);
%      plot(t,mpc3_sim_out.u2_llm3(1:numel(t)),'-','Color','#77AC30','DisplayName','MPC: LLM 3 on Non-Linear Plant','LineWidth',1);
%      plot(t,mpc4_sim_out.u2_llm4(1:numel(t)),'-','Color','#EDB120','DisplayName','MPC: LLM 4 on Non-Linear Plant','LineWidth',1);
% plot(t,fmpc_sim_out.u_llm1_2(1:numel(t)),'g-','DisplayName','FMPC: Linear Local Model 1','LineWidth',1);
% plot(t,fmpc_sim_out.u_llm2_2(1:numel(t)),'b-','DisplayName','FMPC: Linear Local Model 2','LineWidth',1);
% plot(t,fmpc_sim_out.u_llm3_2(1:numel(t)),'c-','DisplayName','FMPC: Linear Local Model 3','LineWidth',1);
% % plot(t,fmpc_sim_out.u_llm4_2(1:numel(t)),'y--','DisplayName','FMPC: Linear Local Model 4','LineWidth',1);
%     plot(t,fmpc_sim_out_ohne_z.u_llm1_2(1:numel(t)),'--','Color','#77AC30','DisplayName','FMPC without Disturbance Suppression: Linear Local Model 1','LineWidth',1);
%     plot(t,fmpc_sim_out_ohne_z.u_llm2_2(1:numel(t)),'--','Color','#0072BD','DisplayName','FMPC without Disturbance Suppression: Linear Local Model 2','LineWidth',1);
%     plot(t,fmpc_sim_out_ohne_z.u_llm3_2(1:numel(t)),'--','Color','#4DBEEE','DisplayName','FMPC without Disturbance Suppression: Linear Local Model 3','LineWidth',1);
    plot(t,fmpc_sim_out.u2(1:numel(t)),'r','DisplayName','FMPC','LineWidth',2);
%         plot(t,fmpc_sim_out_ohne_z.u2(1:numel(t)),'g:','DisplayName','FMPC without Disturbance\newlineSuppression','LineWidth',2);
p = plot(t,u_lim(2,1)+t*0,'k--','DisplayName','Beschränkung','LineWidth',0.5); p.Annotation.LegendInformation.IconDisplayStyle = 'off';
p = plot(t,u_lim(2,2)+t*0,'k--','DisplayName','Beschränkung','LineWidth',0.5); p.Annotation.LegendInformation.IconDisplayStyle = 'off';
title('System Input 2: Motor Generator 1 - Torque T_{MG1}','FontSize',12)
% legend('Location','best');
set(gca,'FontSize',12)

nexttile; hold on; box on; 
% line31 = plot(t,u1(3,1:numel(t)),'y-','DisplayName','MPC: LLM 1 on Linear Plant','LineWidth',1);
% line32 = plot(t,u2(3,1:numel(t)),'y-','DisplayName','MPC: LLM 2 on Linear Plant','LineWidth',1);
% line33 = plot(t,u3(3,1:numel(t)),'y-','DisplayName','MPC: LLM 3 onth Linear Plant','LineWidth',1);
% % plot(t,u4(3,1:numel(t)),'r-','DisplayName','MPC: LLM 4 on Linear Plant','LineWidth',1);
%             plot(t,mpc1_sim_out.u3_llm1(1:numel(t)),'-','Color','#77AC30','DisplayName','MPC (Nc = 15): LLM 1 on Non-Linear Plant','LineWidth',1);
%             plot(t,mpc1_sim_out_Nc5.u3_llm1(1:numel(t)),'-','Color','#7E2F8E','DisplayName','MPC (Nc = 5): LLM 1 on Non-Linear Plant','LineWidth',1);
%             plot(t,mpc1_sim_out_Nc10.u3_llm1(1:numel(t)),'-','Color','#4DBEEE','DisplayName','MPC (Nc = 10): LLM 1 on Non-Linear Plant','LineWidth',1);
%             plot(t,mpc1_sim_out_Nc25.u3_llm1(1:numel(t)),'-','Color','#EDB120','DisplayName','MPC (Nc = 25): LLM 1 on Non-Linear Plant','LineWidth',1);
%             plot(t,mpc1_sim_out_Nc50.u3_llm1(1:numel(t)),'-','Color','#EDB120','DisplayName','MPC (Nc = 25): LLM 1 on Non-Linear Plant','LineWidth',1);
%     plot(t,mpc1_sim_out.u3_llm1(1:numel(t)),'-','Color','#77AC30','DisplayName','MPC: LLM 1 on Non-Linear Plant','LineWidth',1);
%     plot(t,mpc2_sim_out.u3_llm2(1:numel(t)),'-','Color','#77AC30','DisplayName','MPC: LLM 2 on Non-Linear Plant','LineWidth',1);
%     plot(t,mpc3_sim_out.u3_llm3(1:numel(t)),'-','Color','#77AC30','DisplayName','MPC: LLM 3 on Non-Linear Plant','LineWidth',1);
% %     plot(t,mpc4_sim_out.u3_llm4(1:numel(t)),'-','Color','#EDB120','DisplayName','MPC: LLM 4 on Non-Linear Plant','LineWidth',1);
% plot(t,fmpc_sim_out.u_llm1_3(1:numel(t)),'g-','DisplayName','FMPC: Linear Local Model 1','LineWidth',1);
% plot(t,fmpc_sim_out.u_llm2_3(1:numel(t)),'b-','DisplayName','FMPC: Linear Local Model 2','LineWidth',1);
% plot(t,fmpc_sim_out.u_llm3_3(1:numel(t)),'c-','DisplayName','FMPC: Linear Local Model 3','LineWidth',1);
%     plot(t,fmpc_sim_out_ohne_z.u_llm1_3(1:numel(t)),'--','Color','#77AC30','DisplayName','FMPC without Disturbance Suppression: Linear Local Model 1','LineWidth',1);
%     plot(t,fmpc_sim_out_ohne_z.u_llm2_3(1:numel(t)),'--','Color','#0072BD','DisplayName','FMPC without Disturbance Suppression: Linear Local Model 2','LineWidth',1);
%     plot(t,fmpc_sim_out_ohne_z.u_llm3_3(1:numel(t)),'--','Color','#4DBEEE','DisplayName','FMPC without Disturbance Suppression: Linear Local Model 3','LineWidth',1);
% % plot(t,fmpc_sim_out.u_llm4_3(1:numel(t)),'y--','DisplayName','FMPC: Linear Local Model 4','LineWidth',1);
        plot(t,fmpc_sim_out.u3(1:numel(t)),'r','DisplayName','FMPC','LineWidth',2);
%             plot(t,fmpc_sim_out_ohne_z.u3(1:numel(t)),'g:','DisplayName','FMPC without Disturbance\newlineSuppression','LineWidth',2);
p = plot(t,u_lim(3,1)+t*0,'k--','DisplayName','Beschränkung','LineWidth',0.5); p.Annotation.LegendInformation.IconDisplayStyle = 'off';
p = plot(t,u_lim(3,2)+t*0,'k--','DisplayName','Beschränkung','LineWidth',0.5); p.Annotation.LegendInformation.IconDisplayStyle = 'off';
title('System Input 3: Motor Generator 2 - Torque T_{MG2}','FontSize',12)
% legend('Location','best');
xlabel('Time (s)')
set(gca,'FontSize',12)

% nexttile; hold on; box on; 
% line41 = plot(t,ztrack(1:numel(t)),'b','DisplayName','Disturbance Signal','LineWidth',2);
% title('System Disturbance: (reduced) Rotational Mass - Torque T_{Load}','FontSize',12)
% linkaxes(findall(gcf,'type','axes'),'x')
% legend('Location','best');
% xlabel('Time (s)')
% set(gca,'FontSize',12)

% nexttile; hold on; box on; 
% plot(t,fmpc_sim_out.phi_fuzzy1(1:numel(t),1)','g-','DisplayName','\Phi_1','LineWidth',1);
% plot(t,fmpc_sim_out.phi_fuzzy2(1:numel(t),1)','b-','DisplayName','\Phi_2','LineWidth',1);
% plot(t,fmpc_sim_out.phi_fuzzy3(1:numel(t),1)','c-','DisplayName','\Phi_3','LineWidth',1);
% % % % plot(t,fmpc_sim_out.phi_fuzzy4(1:numel(t),1)','y-','DisplayName','\Phi_4','LineWidth',1);
% %     plot(t,fmpc_sim_out_ohne_z.phi_fuzzy1(1:numel(t),1)','--','Color','#77AC30','DisplayName','\Phi_1 (without Disturbance Suppression)','LineWidth',1);
% %     plot(t,fmpc_sim_out_ohne_z.phi_fuzzy2(1:numel(t),1)','--','Color','#0072BD','DisplayName','\Phi_2 (without Disturbance Suppression)','LineWidth',1);
% %     plot(t,fmpc_sim_out_ohne_z.phi_fuzzy3(1:numel(t),1)','--','Color','#4DBEEE','DisplayName','\Phi_3 (without Disturbance Suppression)','LineWidth',1);
% title('Membership Functions ')
% xlabel('Time (s)')
% legend('Location','best');

linkaxes(findall(gcf,'type','axes'),'x')
% lgd = legend(nexttile(2), [line11,line14,line17]);
% lgd.Layout.Tile = 'south';
% lgd = legend(nexttile(2), [line11,line14,line17]);
% lgd = legend(nexttile(2), [line12,line15,line17]);
% lgd = legend(nexttile(2), [line13,line16,line17]);
lgd = legend(nexttile(2), [line17]);
% lgd = legend(nexttile(2), [line17, line19, line110, line111]);
% lgd = legend(nexttile(2), [line17, line18, line41]);
lgd.Layout.Tile = 'east';
lgd.FontSize = 12;
