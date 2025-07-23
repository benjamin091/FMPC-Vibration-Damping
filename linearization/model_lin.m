function [sys_lin,E,A0] = model_lin(num,nxm,nym,nzm,parameter,AP)

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Erstellung des nichtlinearen Systems %%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Init der Symbolic Toolbox
    u = sym('u',[num,1]);
    x = sym('x',[nxm,1]);
    y = sym('y',[nym,1]);
    z = sym('z',[nzm,1]);
    
    ff = sym('ff',[nxm,1]);
    
    syms g_RS r_1 r_2 r_3 J_1 J_2 J_3 J_4 J_p m_p M_1 M_2 M_3 M_4 M_5 d_s c_1 c_2
    
    y(1) = x(2);
    y(2) = x(4);
    
    %% omega_4^1
    ff(1,1) = x(3) - x(4);
    
    ff(2,1) = -(M_2/(M_5*M_1)) * (c_1*x(4)^2+c_2)*x(1) ...
        - (M_2/(M_5*M_1))*d_s*x(3) ...
        + (M_2/(M_5*M_1))*d_s*x(4) ...
        + (1+((M_2*M_3)/(M_1*M_5)))/M_1*u(1) ...
        + ((2*r_1/r_2) + (M_2/M_5)*(2*r_1*M_3/(r_2*M_1) - (r_3/r_2)))/M_1 * u(2) ...
        + M_2/(M_5*M_1)*u(3);

    ff(3,1) = -(1/M_5)*(c_1*x(4)^2+c_2)*x(1) - (1/M_5)*d_s*x(3) + (1/M_5)*d_s*x(4) + ...
        M_3/(M_1*M_5)*u(1) + ((2*r_1 * M_3)/(r_2*M_1)-r_3/r_2)/M_5*u(2) + (1/M_5) * u(3);
    
    ff(4,1) = (1/J_4)*(c_1*x(4)^2*x(1)+c_2*x(1)) + (1/J_4)*d_s*x(3) - (1/J_4)*d_s * x(4) - (1/J_4) * z;


    %% Linearisierung
    A = jacobian(ff,x);
    B = jacobian(ff,u);
    C = jacobian(y,x);
    E = jacobian(ff,z);
    
    g_RS_konst = parameter.g_RS;
    r_1_konst  = parameter.r_1;
    r_2_konst  = parameter.r_2;
    r_3_konst  = parameter.r_3;
    J_1_konst  = parameter.J_1;
    J_2_konst  = parameter.J_2;
    J_3_konst  = parameter.J_3;
    J_4_konst  = parameter.J_4;
    J_p_konst  = parameter.J_p;
    m_p_konst  = parameter.m_p; 
    M_1_konst  = parameter.M_1;
    M_2_konst  = parameter.M_2;
    M_3_konst  = parameter.M_3;
    M_4_konst  = parameter.M_4;
    M_5_konst  = parameter.M_5;
    d_s_konst  = parameter.d_s;
    c_1_konst  = parameter.c_1;
    c_2_konst  = parameter.c_2;
%     x_3_konst  = 0;
    x_1_konst  = 0;
 

%     A_0 = subs(A,[g_RS r_1 J_1 J_2 J_3 J_4 J_p m_p M_1 M_2 M_3 M_4 M_5 d_s c_1 c_2 x(1) x(4)],...
%         [g_RS_konst r_1_konst J_1_konst J_2_konst J_3_konst J_4_konst J_p_konst m_p_konst M_1_konst M_2_konst M_3_konst M_4_konst M_5_konst d_s_konst c_1_konst c_2_konst 0 AP]);
% 
    A_0 = subs(A,[g_RS r_1 r_2 r_3 J_1 J_2 J_3 J_4 J_p m_p M_1 M_2 M_3 M_4 M_5 d_s c_1 c_2],...
        [g_RS_konst r_1_konst r_2_konst r_3_konst J_1_konst J_2_konst J_3_konst J_4_konst J_p_konst m_p_konst M_1_konst M_2_konst M_3_konst M_4_konst M_5_konst d_s_konst c_1_konst c_2_konst]);

    A_0 = subs(A_0,[x(1) x(3) x(4)], [x_1_konst AP AP]);
    
    B_sym = subs(B,[g_RS r_1 r_2 r_3 J_1 J_2 J_3 J_4 J_p m_p M_1 M_2 M_3 M_4 M_5 d_s],...
        [g_RS_konst r_1_konst r_2_konst r_3_konst J_1_konst J_2_konst J_3_konst J_4_konst J_p_konst m_p_konst M_1_konst M_2_konst M_3_konst M_4_konst M_5_konst d_s_konst]);
    
    E_sym = subs(E,[J_4 x(1)],[J_4_konst AP]);
    
    A0 = double(A_0);
    B = double(B_sym);
    C = double(C);
    E = double(E_sym);
    
    sys_lin = ss(A0,B,C,0);
end