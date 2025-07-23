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
% Qy_lmi = Qy;
Qy_lmi = Qy_global;

PhiU_lmi_global = PhiU_global;
F_lmi_global = F_global;
H_lmi = H_global; 

%% Matrizen des geschlossenen RK mit MPC
Kmpc_lmi_global = zeros(nu_lmi*n_mod_lmi,nx_lmi);
Acl_lmi_global = zeros(nx_lmi*n_mod_lmi,nx_lmi*n_mod_lmi);


for i = 1:n_mod_lmi
    Kmpc_lmi_global(nu_lmi*(i-1)+1:nu_lmi*i,:) = [eye(nu_lmi) zeros(nu_lmi,(Nc_lmi-1)*nu_lmi)]...
        *H_lmi((i-1)*Nc_lmi*nu_lmi+1:i*Nc_lmi*nu_lmi,:)^(-1)...
        *PhiU_lmi_global((i-1)*Np_lmi*ny_lmi+1:i*Np_lmi*ny_lmi,:)'...
        *Qy_lmi(1+(i-1)*Np_lmi*ny_lmi:i*Np_lmi*ny_lmi,:)*F_lmi_global( (i-1)*Np_lmi*ny_lmi+1 : i*Np_lmi*ny_lmi,:);

%    Kmpc_lmi_global(nu_lmi*(i-1)+1:nu_lmi*i,:) = [eye(nu_lmi) zeros(nu_lmi,(Nc_lmi-1)*nu_lmi)]*...
%         H_lmi((i-1)*Nc_lmi*nu_lmi+1:i*Nc_lmi*nu_lmi,:)^-1*PhiU_lmi_global((i-1)*Np_lmi*ny_lmi+1:i*Np_lmi*ny_lmi,:)'...
%         *F_lmi_global( (i-1)*Np_lmi*ny_lmi+1 : i*Np_lmi*ny_lmi,:);
end


%% Linear Matrix Inequality - 2D
P_global_qdr = zeros(nx_lmi*n_mod_lmi,nx_lmi*n_mod_lmi);
P_global_lin = zeros(nx_lmi*n_mod_lmi^2,nx_lmi);

Acl_lmi_global_lin = zeros(nx_lmi*n_mod_lmi^2,nx_lmi);

eig_lmi_global = zeros(nx_lmi*n_mod_lmi,n_mod_lmi);
eig_lmi_val = zeros(n_mod_lmi^3*nx_lmi,n_mod_lmi);

for ii = 1:n_mod_lmi
    for iii = 1:n_mod_lmi
        Acl_lmi_global(nx_lmi*(iii-1)+1:nx_lmi*iii,nx_lmi*(ii-1)+1:nx_lmi*ii) =  A_lmi_global(nx_lmi*(iii-1)+1:nx_lmi*iii,:) - B*Kmpc_lmi_global(nu_lmi*(ii-1)+1:nu_lmi*ii,:);

        setlmis([]) 
            [p,n,sp] = lmivar(1,[6 1]);
%             S1 = newlmi;
            lmiterm([1 1 1 p],Acl_lmi_global(nx_lmi*(iii-1)+1:nx_lmi*iii,nx_lmi*(ii-1)+1:nx_lmi*ii)',Acl_lmi_global(nx_lmi*(iii-1)+1:nx_lmi*iii,nx_lmi*(ii-1)+1:nx_lmi*ii))
            lmiterm([1 1 1 p],-1,1)
        lmis = getlmis;
        
        [tmin,xfeas] = feasp(lmis);

        P_global_qdr(nx_lmi*(iii-1)+1:nx_lmi*iii,nx_lmi*(ii-1)+1:nx_lmi*ii) = dec2mat(lmis,xfeas,p);
        eig_lmi_global(nx_lmi*(iii-1)+1:nx_lmi*iii,ii) = eig(P_global_qdr(nx_lmi*(iii-1)+1:nx_lmi*iii,nx_lmi*(ii-1)+1:nx_lmi*ii));
    end
end

%% Suche nach der P-Matrix, für die alle LMIs erfüllt sind
P_global_lin(1:nx_lmi*n_mod_lmi,:) = P_global_qdr(1:nx_lmi*n_mod_lmi,1:nx_lmi);

% Acl_lmi_global_lin(1:nx_lmi*n_mod_lmi,:) = P_global_qdr(1:nx_lmi*n_mod_lmi,1:nx_lmi);

for k = 2:n_mod_lmi
    P_global_lin(nx_lmi*n_mod_lmi*(k-1)+1:k*nx_lmi*n_mod_lmi,:) = P_global_qdr(:,(k-1)*nx_lmi+1:nx_lmi*k);
end


for j = 1:n_mod_lmi^2
    for jj = 1:n_mod_lmi
        for jjj = 1:n_mod_lmi
            eig_lmi_val(nx_lmi*(jjj-1)+1+(j-1)*nx_lmi*n_mod_lmi:nx_lmi*jjj+(j-1)*nx_lmi*n_mod_lmi,jj) = ...
                eig(Acl_lmi_global(nx_lmi*(jjj-1)+1:nx_lmi*jjj,nx_lmi*(jj-1)+1:nx_lmi*jj)'...
                *P_global_lin(nx_lmi*(j-1)+1:nx_lmi*j,:)...
                *Acl_lmi_global(nx_lmi*(jjj-1)+1:nx_lmi*jjj,nx_lmi*(jj-1)+1:nx_lmi*jj) ...
                - P_global_lin(nx_lmi*(j-1)+1:nx_lmi*j,:));
        end
    end
end



% neg_def = eig_lmi_val < 0;
% 
% for kk = 1:nx_lmi*l
%     = neg_def(nx_lmi*nx_lmi*l*(kk-1)+1:nx_lmi*nx_lmi*l*kk)
% end
% 
% 
definitheit_pos = eig_lmi_val < 0;








