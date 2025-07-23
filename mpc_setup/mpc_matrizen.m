function [F,PhiU,PhiZ,H] = mpc_matrizen(A,B,C,E,Np,Nc,Qy,Ru)

    [nx,nu] = size(B);  
    ny = size(C,1);
    nz = size(E,2);
    
    F = zeros(Np*ny,nx);

    for i = 1:Np
        row = (1:ny)+(i-1)*ny;
        F(row,:) = C*A^i;
    end

    PhiU = zeros(Np*ny,Nc*nu); 
    col = zeros(Np*ny,nu); 
    zero = zeros(size(C*B));

    for i = 1:Np
        row = (1:ny) + (i-1)*ny;
        col(row,:) = C*A^(i-1)*B;
    end

    for i=1:Nc
        row = (1:nu) + (i-1)*nu;
        PhiU( : , row ) = col;
        col = [ zero ; col(1:end-ny,:) ];
    end
    
    PhiZ = zeros(Np*ny,Np*nz); 
    col = zeros(Np*ny,nz); 
    zero = zeros(size(C*E)); 

    for i = 1:Np
        row = (1:ny) + (i-1)*ny;
        col(row,:) = C*A^(i-1)*E;
    end
    
    for i = 1:Np
        row = (1:nz) + (i-1)*nz;
        PhiZ(:,row) = col;
        col = [zero;col(1:end-ny,:)];
    end
    
    H = PhiU' * Qy * PhiU + Ru;
end