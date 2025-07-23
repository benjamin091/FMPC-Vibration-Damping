%% (erweiterte) Systemmatrizen
ny_lmi = ny;
nu_lmi = nu;
nx_lmi = nx;

% n_mod_lmi = n_modelle_ende;
n_mod_lmi = l;

A_lmi_global = A_global;
B_lmi = B;

%% MPC-Matrizen
Np_lmi = Np;
Nc_lmi = Nc;
Qy_lmi = Qy;
% Qy_lmi = Qy_global;

PhiU_lmi_global = PhiU_global;
F_lmi_global = F_global;
H_lmi = H_global; 

%% Matrizen des geschlossenen RK mit MPC
Kmpc_lmi_global = zeros(nu_lmi*n_mod_lmi,nx_lmi);
Acl_lmi_global_einzeln = zeros(nx_lmi*n_mod_lmi,nx_lmi);
Acl_lmi_global = zeros(nx_lmi*n_mod_lmi,nx_lmi*n_mod_lmi);



%% Berechnung der MPC-Verstärkung
for i = 1:n_mod_lmi
    Kmpc_lmi_global(nu_lmi*(i-1)+1:nu_lmi*i,:) = [eye(nu_lmi) zeros(nu_lmi,(Nc_lmi-1)*nu_lmi)]...
        *H_lmi((i-1)*Nc_lmi*nu_lmi+1:i*Nc_lmi*nu_lmi,:)^(-1)...
        *PhiU_lmi_global((i-1)*Np_lmi*ny_lmi+1:i*Np_lmi*ny_lmi,:)'...
        *Qy_lmi*F_lmi_global( (i-1)*Np_lmi*ny_lmi+1 : i*Np_lmi*ny_lmi,:);
end

%% Überprüfung der Eigenwerte des CL-Systems
%% Einzelne MPC
eig_cl_einzeln = zeros(nx_lmi*n_mod_lmi,1);

for iii = 1:n_mod_lmi
    Acl_lmi_global_einzeln(nx_lmi*(iii-1)+1:nx_lmi*iii,:) =  A_lmi_global(nx_lmi*(iii-1)+1:nx_lmi*iii,:) - B*Kmpc_lmi_global(nu_lmi*(iii-1)+1:nu_lmi*iii,:);
    eig_cl(nx_lmi*(iii-1)+1:nx_lmi*iii,:) = eig(Acl_lmi_global_einzeln(nx_lmi*(iii-1)+1:nx_lmi*iii,:));
end

%% Permutierte MPC
eig_cl_permutiert = zeros(nx_lmi*n_mod_lmi,1*n_mod_lmi);

for ii = 1:n_mod_lmi
    for iii = 1:n_mod_lmi
        Acl_lmi_global(nx_lmi*(iii-1)+1:nx_lmi*iii,nx_lmi*(ii-1)+1:nx_lmi*ii) =  A_lmi_global(nx_lmi*(iii-1)+1:nx_lmi*iii,:) - B*Kmpc_lmi_global(nu_lmi*(ii-1)+1:nu_lmi*ii,:);
        eig_cl_permutiert(nx_lmi*(iii-1)+1:nx_lmi*iii,ii) = eig(Acl_lmi_global(nx_lmi*(iii-1)+1:nx_lmi*iii,nx_lmi*(ii-1)+1:nx_lmi*ii));
    end
end


%% Linear Matrix Inequality - 2D
P_global_qdr = zeros(nx_lmi*n_mod_lmi,nx_lmi*n_mod_lmi);
P_global_lin = zeros(nx_lmi*n_mod_lmi^2,nx_lmi);

Acl_lmi_global_lin = zeros(nx_lmi*n_mod_lmi^2,nx_lmi);

eig_lmi_global = zeros(nx_lmi*n_mod_lmi,n_mod_lmi);

%% LMI-System
        setlmis([]) 
            [p,n,sp] = lmivar(1,[6 1]);
            lmiterm([1 1 1 p],Acl_lmi_global(1:6,1:6)',Acl_lmi_global(1:6,1:6))
            lmiterm([1 1 1 p],-1,1)
            lmiterm([2 1 1 p],Acl_lmi_global(7:12,1:6)',Acl_lmi_global(7:12,1:6))
            lmiterm([2 1 1 p],-1,1)
            lmiterm([3 1 1 p],Acl_lmi_global(13:18,1:6)',Acl_lmi_global(13:18,1:6))
            lmiterm([3 1 1 p],-1,1)
            lmiterm([4 1 1 p],Acl_lmi_global(1:6,7:12)',Acl_lmi_global(1:6,7:12))
            lmiterm([4 1 1 p],-1,1)
            lmiterm([5 1 1 p],Acl_lmi_global(7:12,7:12)',Acl_lmi_global(7:12,7:12))
            lmiterm([5 1 1 p],-1,1)
            lmiterm([6 1 1 p],Acl_lmi_global(13:18,7:12)',Acl_lmi_global(13:18,7:12))
            lmiterm([6 1 1 p],-1,1)
            lmiterm([7 1 1 p],Acl_lmi_global(1:6,13:18)',Acl_lmi_global(1:6,13:18))
            lmiterm([7 1 1 p],-1,1)
            lmiterm([8 1 1 p],Acl_lmi_global(7:12,13:18)',Acl_lmi_global(7:12,13:18))
            lmiterm([8 1 1 p],-1,1)
            lmiterm([9 1 1 p],Acl_lmi_global(13:18,13:18)',Acl_lmi_global(13:18,13:18))
            lmiterm([9 1 1 p],-1,1)
        lmis = getlmis;

        %% feasp-Solver
        options = zeros(1,5);
        options(1) = 0;
        options(2) = 0; %max. Anzahl an Iteration; default 100
        options(3) = 0; % feasibility-Radius; für > 0 höhere Beschränkung auf P
        options(4) = 0; % Geschwindigkeit; für < 10 rechnet Solver schneller aber ungenauer
        options(5) = 0; %
        target = -1;

        [tmin,xfeas] = feasp(lmis,options,target);
        P = dec2mat(lmis,xfeas,p);

        evlmi = evallmi(lmis,xfeas);
        [lhs,rhs] = showlmi(evlmi,1);
        eig(lhs-rhs)

        %% mincx-Solver
%         options = zeros(1,5);
%         options(1) = 0; % gewünschte Genauigkeit; default 10-2
%         options(2) = 0; % max. Anzahl an Iteration; default 100
%         options(3) = 0; % feasibility-Radius; für > 0 höhere Beschränkung auf P
%         options(4) = 0; % Geschwindigkeit; für < 10 rechnet Solver schneller aber ungenauer
%         options(5) = 0; %
%         
%         c = ones(21,21);
%         xinit = ones(1,21);
%         target = 0;
% 
%         [copt,xopt] = mincx(lmis,c);
%         [copt,xopt] = mincx(lmis,c,options,xinit,target);
%         P = dec2mat(lmis,xopt,p);
        %% gevp-Solver



        %% Auswertung
        eig_P = eig(P);


%% Suche nach der P-Matrix, für die alle LMIs erfüllt sind


eig_lmi_val = zeros(n_mod_lmi*nx_lmi,n_mod_lmi);


    for jj = 1:n_mod_lmi
        for jjj = 1:n_mod_lmi
            eig_lmi_val(nx_lmi*(jjj-1)+1:nx_lmi*jjj,jj) = ...
                eig(Acl_lmi_global(nx_lmi*(jjj-1)+1:nx_lmi*jjj,nx_lmi*(jj-1)+1:nx_lmi*jj)'*P*Acl_lmi_global(nx_lmi*(jjj-1)+1:nx_lmi*jjj,nx_lmi*(jj-1)+1:nx_lmi*jj) - P);
        end
    end

eig_lmi_val_neu = zeros(nx_lmi,n_mod_lmi^2);
eig_lmi_val_neu(1:nx_lmi,1:n_mod_lmi) = eig_lmi_val(1:nx_lmi,1:n_mod_lmi);

    for k = 2:n_mod_lmi
        eig_lmi_val_neu(:,n_mod_lmi*(k-1)+1:k*n_mod_lmi) = eig_lmi_val((k-1)*nx_lmi+1:nx_lmi*k,:);
    end

P_table = array2table(P);
filename = 'Lyapunov_P.xlsx';
writetable(P_table,filename);

P_eig_table = array2table(eig_lmi_val_neu,'VariableNames',{'EW - LMI 1','EW - LMI 2','EW - LMI 3','EW - LMI 4','EW - LMI 5','EW - LMI 6','EW - LMI 7','EW - LMI 8','EW - LMI 9'});
filename = 'Lyapunov_P_eig.xlsx';
writetable(P_eig_table,filename);



