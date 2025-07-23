    AB = linspace(-1,31,1000);
%     b = 0; %für b = 0 ... Dreieck; für b > 0 ... Trapez

    %% 3x Modelle ----------------------
      phi_fuzzy1 = trapmf(AB,[-100000000,-100000000,fuzzy_menge(1,1)+b,fuzzy_menge(1,2)-b]);
      phi_fuzzy2 = trapmf(AB,[fuzzy_menge(1,1)+b,fuzzy_menge(1,2)-b,fuzzy_menge(1,2)+b,fuzzy_menge(2,2)-b]);
      phi_fuzzy3 = trapmf(AB,[fuzzy_menge(1,2)+b,fuzzy_menge(2,2)-b,100000000,100000000]);

    figure
    plot(AB,phi_fuzzy1,'g-','LineWidth',2)
    hold on
    plot(AB,phi_fuzzy2,'b-','LineWidth',2)
    plot(AB,phi_fuzzy3,'c-','LineWidth',2)
    xline([fuzzy_menge(1,1),fuzzy_menge(2,1),fuzzy_menge(2,2)],'k-',{'LLM1', 'LLM2', 'LLM3'},'LabelVerticalAlignment','middle','LabelHorizontalAlignment','left','LineWidth',2)
%     xline([fuzzy_menge(1,1),fuzzy_menge(1,2)-b,fuzzy_menge(1,2)+b,fuzzy_menge(2,2)],'k--')
    xline([0,29.154],'r--',{'Unteres Ende des Arbeitsbereichs','Oberes Ende des Arbeitsbereichs'},'LabelVerticalAlignment','middle','LineWidth',2)
%     title('Membership Functions')
    xlabel('Winkelgeschwindigkeit \omega_4 [1/s]')
    set(gca,'FontSize',12)

    