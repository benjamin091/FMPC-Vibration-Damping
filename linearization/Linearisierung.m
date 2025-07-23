function [Am_lin_global,Am_lin_global_val,Bm_lin,Cm_lin,Em_lin,nugap_final_val,AP_LLM] = Linearisierung(omega4_range,parameter,nxm,num,nym,nzm,stueckelung,gap_schwell_min,gap_schwell_max)
    %% Bestimmung der nugap-Partitionierung
    [nugap_final,omega4_band_final] = nugap_metrik_distanz(num,nxm,nym,nzm,omega4_range,stueckelung,gap_schwell_min,gap_schwell_max,parameter);
    
    %% Bestimmung der lokalen linearen Modelle und Systemanalyse
    AP = omega4_band_final;
    
    sys_final_c = cell(length(omega4_band_final),2);
    Am_lin_global = zeros(nxm*length(omega4_band_final),nxm);
    
    eigen_c = cell(length(omega4_band_final),1);
    steuerbarkeit = zeros(length(omega4_band_final),1);
    beobachtbarkeit = zeros(length(omega4_band_final),1);
    
    for i = 1:length(omega4_band_final)
        %% lokalen linearen Modelle
        [sys_lin,E] = model_lin(num,nxm,nym,nzm,parameter,AP(i));
    
        sys_final_c{i,1} = sys_lin;
        sys_final_c{i,2} = E;
        
        Am_lin_global((i-1)*nxm+1:i*nxm,:) = sys_lin.A;
        Bm_lin = sys_lin.B;
        Cm_lin = sys_lin.C;
        Em_lin = E;
    
        %% Eigenwerte
        eigen_c{i,1} = eig(sys_final_c{i,1});
        %% Steuerbarkeit
        steuerbarkeit(i,1) = rank(ctrb(sys_final_c{i,1}.A,sys_final_c{i,1}.B));
        %% Beobachtbarkeit
        beobachtbarkeit(i,1) = rank(obsv(sys_final_c{i,1}.A,sys_final_c{i,1}.C));
    end
    
    %% Validierung nugap
    n = 3; %Anzahl der Modelle
    n1 = 4;
    n2 = 1;
    m = (length(sys_final_c)-(1+n1+n2))/(n-1); %Abstand zw den Modellen
    
    sys_val_c = cell(n,2);
    nugap_final_val = zeros(length(sys_val_c)-1,5);
    AP_LLM = zeros(1,n);
    
    sys_val_c{1,1} = sys_final_c{1+n1,1};
    sys_val_c{2,1} = sys_final_c{1+n1+m,1};
    sys_val_c{3,1} = sys_final_c{1+n1+2*m,1};
%     sys_val_c{4,1} = sys_final_c{1+n1+3*m,1};
    % sys_val_c{5,1} = sys_final_c{1+n1+4*m,1};

    sys_val_c{1,2} = E;
    
    nugap_final_val(1,2) = nugap_final(1+n1,2);
    nugap_final_val(1,4) = nugap_final(1+n1,4);
    nugap_final_val(1,3) = nugap_final(m+n1,3);
    nugap_final_val(1,5) = nugap_final(m+n1,5);
    
    nugap_final_val(2,2) = nugap_final(1+n1+m,2);
    nugap_final_val(2,4) = nugap_final(1+n1+m,4);
    nugap_final_val(2,3) = nugap_final(2*m+n1,3);
    nugap_final_val(2,5) = nugap_final(2*m+n1,5);
    
%     nugap_final_val(3,2) = nugap_final(1+n1+2*m,2);
%     nugap_final_val(3,4) = nugap_final(1+n1+2*m,4);
%     nugap_final_val(3,3) = nugap_final(3*m+n1,3);
%     nugap_final_val(3,5) = nugap_final(3*m+n1,5);
    
    % nugap_final_val(4,2) = nugap_final(1+n1+3*m,2);
    % nugap_final_val(4,4) = nugap_final(1+n1+3*m,4);
    % nugap_final_val(4,3) = nugap_final(4*m+n1,3);
    % nugap_final_val(4,5) = nugap_final(4*m+n1,5);
    % 
    
    
    % sys_val_c{1,1} = sys_final_c{1,1};
    % sys_val_c{2,1} = sys_final_c{1+m,1};
    % sys_val_c{3,1} = sys_final_c{1+2*m,1};
    % sys_val_c{4,1} = sys_final_c{1+3*m,1};
    % 
    % nugap_final_val(1,2) = nugap_final(1,2);
    % nugap_final_val(1,4) = nugap_final(1,4);
    % nugap_final_val(1,3) = nugap_final(m,3);
    % nugap_final_val(1,5) = nugap_final(m,5);
    % 
    % nugap_final_val(2,2) = nugap_final(1+m,2);
    % nugap_final_val(2,4) = nugap_final(1+m,4);
    % nugap_final_val(2,3) = nugap_final(2*m,3);
    % nugap_final_val(2,5) = nugap_final(2*m,5);
    % 
    % nugap_final_val(3,2) = nugap_final(1+2*m,2);
    % nugap_final_val(3,4) = nugap_final(1+2*m,4);
    % nugap_final_val(3,3) = nugap_final(3*m,3);
    % nugap_final_val(3,5) = nugap_final(3*m,5);
    
    for ii = 1:n
        sys_lin_val = sys_val_c{ii,1};
        Am_lin_global_val((ii-1)*nxm+1:ii*nxm,:) = sys_lin_val.A;
        Bm_lin = sys_lin.B;
        Cm_lin = sys_lin.C;
        Em_lin = E;
    end
    
    AP_LLM(1,1) = nugap_final_val(1,4);
    for iii = 1:n-1
        [nugap_final_val(iii,1),~] = gapmetric(sys_val_c{iii,1},sys_val_c{iii+1,1});
        AP_LLM(iii+1) = nugap_final_val(iii,5);
    end


    %% Excel-File
    nugap_final_table = array2table([[1:length(nugap_final)]', [2:length(nugap_final)+1]', nugap_final],'VariableNames',{'AP_i', 'AP_{i+1}', 'nugap','k(omega4_i)','k(omega4_{i+1})','omega4_i','omega4_{i+1}'});
    filename = 'Partitionierung.xlsx';
    writetable(nugap_final_table,filename);
end