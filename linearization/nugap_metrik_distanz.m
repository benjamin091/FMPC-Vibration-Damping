function [nugap_final,omega4_band_final] = nugap_metrik_distanz(num,nxm,nym,nzm,omega4_range,stueckelung,gap_schwell_min,gap_schwell_max,parameter)

%% Bestimmung der nu-Gap anhand der anfänglich gewählten Partitionierung (= stueckelung)
delta_omega4 = (max(omega4_range) - min(omega4_range))/stueckelung;
omega4_band = [omega4_range(1):delta_omega4:omega4_range(2)]';
AP = omega4_band;

%% Initialisierung der nugap, gap und der Systeme
sys_gap = cell(length(omega4_band),2);
nugap = zeros(length(omega4_band)-1,1);

%% Auffüllen des System-Arrays und der gap-Arrays anhand der anfängliche Partitionierung
for i = 1:length(omega4_band)
%     parameter=init_parameter_nl(); 
%     [sys_lin,~] = model_lin(num,nxm,nym,nzm,parameter,AP(i));
    [sys_lin,~,A0] = model_lin(num,nxm,nym,nzm,parameter,AP(i));
    sys_gap{i,1} = sys_lin;
    sys_gap{i,2} = omega4_band(i);
    sys_gap{i,3} = A0;
end

for i = 1:length(omega4_band)-1
    [~,nugap(i,1)] = gapmetric(sys_gap{i,1},sys_gap{i+1,1});
    nugap(i,2) = omega4_band(i);
    nugap(i,3) = omega4_band(i+1);
end

%% Initialisierung für die Korrekturalgorithmen
zaehler1 = 1;
zaehler2 = 1;

nugap_col1 = nugap(:,1);
nugap_col23 = nugap(:,2:3);

nugap_col1_vor = zeros(size(nugap_col1));
nugap_col23_vor = zeros(size(nugap_col23));

nugap_col23_aktuell_min = zeros(2,2);
nugap_col1_aktuell_max_vor = zeros(2,1);
nugap_col23_aktuell_max_nach = zeros(2,2);


%% Korrekturalgorithmus für die Partitionierung anhand der nugap-Metrik
min_gap_col1 = min(nugap_col1(2:end-1,1));
index_min = find(nugap_col1 == min_gap_col1);

    %% Im Falle zu feiner angenommener Initialpartitionierung - Verringerung der Modellanzahl
%     while min(nugap_col1(2:end-1,1)) < gap_schwell_min 
    while min_gap_col1 < gap_schwell_min
        %% Zwei Modellpunkte werden zu einem vereint
        omega4_1 = nugap_col23(index_min,1); %Einträge von k sind streng monoton fallend 
        omega4_2 = nugap_col23(index_min,2);
        omega_neu = (omega4_1 + omega4_2)/2;
    
%         parameter = init_parameter_nl; 
        [sys_lin,~,~] = model_lin(num,nxm,nym,nzm,parameter,omega_neu);
        sys_kor = sys_lin;
       
%         parameter = init_parameter_nl; 
        [sys_lin,~,~] = model_lin(num,nxm,nym,nzm,parameter,nugap_col23(index_min-1,1));
        sys_kor_vor = sys_lin;
    
%         parameter = init_parameter_nl; 
        [sys_lin,~,~] = model_lin(num,nxm,nym,nzm,parameter,nugap_col23(index_min+1,2));
        sys_kor_nach = sys_lin;
        
        %% Entnahme der nugap-Werte vor dem Gefundenen
        nugap_col1_vor(1:index_min-2,1) = nugap_col1(1:index_min-2,1);
        nugap_col23_vor(1:index_min-2,:) = nugap_col23(1:index_min-2,:);
    
        %% Update des nugap-Wertes
        [~,nugap_kor1] = gapmetric(sys_kor_vor,sys_kor); % gap vor index wird upgedated
        %gap mit index wird gelöscht
        [~,nugap_kor2] = gapmetric(sys_kor,sys_kor_nach); % gap nach index wird upgedated
        
        nugap_col23_aktuell_min(1,1:2) = [nugap_col23(index_min-1,1), omega_neu];
        nugap_col23_aktuell_min(2,1:2) = [omega_neu, nugap_col23(index_min+1,2)];
        
        %% Entnahme der nugap-Werte vor dem Gefundenen
        if index_min <= length(nugap_col1)-1
            nugap_col1_nach = nugap_col1(index_min+2:end);
            nugap_col1 = [nugap_col1_vor(1:index_min-2,1); nugap_kor1; nugap_kor2; nugap_col1_nach];
    
        else %% falls minimaler nugap-Wert der letzte Wert ist, wird hinten nichts drangehängt
            nugap_col1 = [nugap_col1_vor(1:index_min-2,1); nugap_kor1; nugap_kor2];
        end
        
        if index_min <= length(nugap_col23)
            nugap_col23_nach = nugap_col23(index_min+2:end,:);
            nugap_col23 = [nugap_col23_vor(1:index_min-2,:); nugap_col23_aktuell_min; nugap_col23_nach];
        else %% falls minimaler nugap-Wert der letzte Wert ist, wird hinten nichts drangehängt
            nugap_col23 = [nugap_col23_vor(1:index_min-2,:); nugap_col23_aktuell_min];
        end
        zaehler1 = zaehler1 + 1; %% Kontrollzähler


        %% Welchen Index hat der niedrigste nugap-Wert
        min_gap_col1 = min(nugap_col1(2:end-1,1));
        index_min = find(nugap_col1 == min_gap_col1);

    end
nugap_kontroll(:,1) = nugap_col1; %Zwischenergebnis zu Kontrolle bei zu feiner Stückelung
nugap_kontroll(:,2:3) = nugap_col23; 
    

    %% Funktionsweise analog zu oben
%     while max(nugap_col1(1:end,1)) > gap_schwell_max 
% 
%         max_gap_col1 = max(nugap_col1(1:end,1));
%         index_max = find(nugap_col1 == max_gap_col1);

max_gap_col1 = max(nugap_col1(1:end,1));
index_max = find(nugap_col1 == max_gap_col1);

     while max_gap_col1 > gap_schwell_max 

        omega4_1 = nugap_col23(index_max,1); %Einträge von k sind streng monoton steigend 
        omega4_2 = nugap_col23(index_max,2);
        omega_neu = (omega4_2 - omega4_1)/2 + omega4_1;
        
%         parameter = init_parameter_nl; 
        [sys_lin,~,~] = model_lin(num,nxm,nym,nzm,parameter,omega_neu);
        sys_kor = sys_lin;
       
%         parameter = init_parameter_nl; 
        [sys_lin,~,~] = model_lin(num,nxm,nym,nzm,parameter,nugap_col23(index_max,1));
        sys_kor_vor = sys_lin;
    
%         parameter = init_parameter_nl; 
        [sys_lin,~,~] = model_lin(num,nxm,nym,nzm,parameter,nugap_col23(index_max,2));
        sys_kor_nach = sys_lin;
    
        nugap_col1_vor(1:index_max-1,1) = nugap_col1(1:index_max-1,1);
        nugap_col23_vor(1:index_max-1,:) = nugap_col23(1:index_max-1,:);
    
        [~,nugap_kor1] = gapmetric(sys_kor_vor,sys_kor); 
        [~,nugap_kor2] = gapmetric(sys_kor,sys_kor_nach); 
        
        nugap_col1_aktuell_max_vor(1:2,1) = [nugap_kor1;nugap_kor2];
    
        nugap_col23_aktuell_max_nach(1,1:2) = [nugap_col23(index_max,1), omega_neu];
        nugap_col23_aktuell_max_nach(2,1:2) = [omega_neu, nugap_col23(index_max,2)];
    
        if index_max < length(nugap_col1)
            nugap_col1_nach = nugap_col1(index_max+1:end);
            nugap_col1 = [nugap_col1_vor(1:index_max-1,1); nugap_col1_aktuell_max_vor; nugap_col1_nach];
        elseif index_max == length(nugap_col1)
            nugap_col1 = [nugap_col1_vor(1:index_max-1,1); nugap_col1_aktuell_max_vor];
        end
        
        if index_max < length(nugap_col23)
            nugap_col23_nach = nugap_col23(index_max+1:end,:);
            nugap_col23 = [nugap_col23_vor(1:index_max-1,:); nugap_col23_aktuell_max_nach; nugap_col23_nach];
        elseif index_max == length(nugap_col23)
            nugap_col23 = [nugap_col23_vor(1:index_max-1,:); nugap_col23_aktuell_max_nach];
        end
        zaehler2 = zaehler2 + 1;
        %% Welchen Index hat der höchste nugap-Wert
        max_gap_col1 = max(nugap_col1(1:end,1));
        index_max = find(nugap_col1 == max_gap_col1);
    end

    nugap_final(:,1) = nugap_col1;
%     nugap_final(:,2:3) = nugap_col23;
%     c_1 = 10;
%     c_2 = 0.07;
    nugap_final(:,2) = parameter.c_1*nugap_col23(:,1).^2+parameter.c_2;
    nugap_final(:,3) = parameter.c_1*nugap_col23(:,2).^2+parameter.c_2;
    nugap_final(:,4:5) = nugap_col23;
    omega4_band_final(:,1) = nugap_final(:,4);
    omega4_band_final(end+1,1) = nugap_final(end,5);
end