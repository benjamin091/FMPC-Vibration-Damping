function [A,B,C,E] = mpc_modellerweiterung(Am,Bm,Cm,Em)
%     Am = sys.A;
%     Bm = sys.B; 
%     Cm = sys.C;
    

    nxm = size(Bm,1); 
    nym = size(Cm, 1); 

    Om = zeros(nym,nxm);    
              
    A = [Am,Om';Cm*Am,eye(nym)];
    B = [Bm;Cm*Bm];
    C = [Om,eye(nym)];
    E = [Em;Cm*Em];
end