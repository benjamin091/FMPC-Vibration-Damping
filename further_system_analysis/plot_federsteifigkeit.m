%% Feder-Steifigkeit

    omega4 = [0 30];
    omega4_range = omega4(1):omega4(2);
    c_1 = 0.2;
    c_2 = 50;
    
    k_nl = c_1*omega4_range.^2+c_2;

    %% 3x Modelle
%     figure
%     plot(omega4_range,k_nl)
%     xline(0,'k-',{'f_{min} =~ 2.4 Hz'},'LabelVerticalAlignment','middle','LabelHorizontalAlignment','left','LineWidth',2);
%     xline(29.154,'k-',{'f_{max} =~ 5 Hz'},'LabelVerticalAlignment','middle','LabelHorizontalAlignment','right','LineWidth',2);
%     xline([AP_LLM(1),AP_LLM(2),AP_LLM(3)],'k--',{'Operation Point 1','Operation Point 2','Operation Point 3'},'LabelVerticalAlignment','middle','LabelHorizontalAlignment','left','LineWidth',1);
%     xlim([-5, 35])
%     xlabel('Rotational Speed \omega_4 [1/s]');
%     ylabel('Spring Parameter k_s [kgm/s²]');
%     title('Non-linear characteristic curve of the spring parameter k_s(\omega_4) = 0.2 \omega_4^2 + 50');

    %% 4x Modelle
%     figure
%     plot(omega4_range,k_nl)
%     xline(0,'k-',{'f_{min} =~ 2.4 Hz'},'LabelVerticalAlignment','middle','LabelHorizontalAlignment','left','LineWidth',2);
%     xline(29.154,'k-',{'f_{max} =~ 5 Hz'},'LabelVerticalAlignment','middle','LabelHorizontalAlignment','right','LineWidth',2);
%     xline([AP_LLM(2),AP_LLM(3),AP_LLM(4)],'k--',{'Operation Point 2','Operation Point 3','Operation Point 4'},'LabelVerticalAlignment','middle','LabelHorizontalAlignment','left','LineWidth',1);
%     xline([AP_LLM(1)],'r--',{'Operation Point 1'},'LabelVerticalAlignment','middle','LabelHorizontalAlignment','left','LineWidth',1);
%     xlim([-5, 35])
%     xlabel('Rotational Speed \omega_4 [1/s]');
%     ylabel('Spring Parameter k_s [kgm/s²]');
%     title('Non-linear characteristic curve of the spring parameter k_s(\omega_4) = 0.2 \omega_4^2 + 50');

    %% 4x Modelle red
    figure
    plot(omega4_range,k_nl,'LineWidth',2)
    xline(0,'k-',{'f_{min} =~ 2.4 Hz'},'LabelVerticalAlignment','middle','LabelHorizontalAlignment','left','LineWidth',2,'FontSize',12);
    xline(29.154,'k-',{'f_{max} =~ 5 Hz'},'LabelVerticalAlignment','middle','LabelHorizontalAlignment','right','LineWidth',2,'FontSize',12);
    xline([AP_LLM(1),AP_LLM(2),AP_LLM(3)],'k--',{'Arbeitspunkt 1','Arbeitspunkt 2','Arbeitspunkt 3'},'LabelVerticalAlignment','middle','LabelHorizontalAlignment','left','LineWidth',2,'FontSize',12);
%     xline([AP_LLM(1)],'r--',{'Operation Point 1'},'LabelVerticalAlignment','middle','LabelHorizontalAlignment','left','LineWidth',1);
    xlim([-5, 35])
    xlabel('Winkelgeschwindigkeit \omega_4 [1/s]','FontSize',12);
    ylabel('Federparameter k_s [kgm/s²]','FontSize',12);
%     title('Non-linear characteristic curve of the spring parameter k_s(\omega_4) = 0.2 \omega_4^2 + 50');
    set(gca,'FontSize',12)