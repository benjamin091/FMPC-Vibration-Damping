%% Bodeplot
%% Zuerst Linearisierung_main laden (!)

figure %zeitkontinuierliches System
for i = 1:(length(Am_lin_global_val)/nxm)
%     Am = sys_final_c{i,1}.A;
%     Bm = sys_final_c{i,1}.B;
%     Am = Am_lin_global((i-1)*nxm+1:i*nxm,:);
%     Bm = Bm_lin_global((i-1)*nxm+1:i*nxm,:);
    Am = Am_lin_global_val((i-1)*nxm+1:i*nxm,:);
    Bm = Bm_lin;

    sys = ss(Am,Bm,[0 0 0 1],[]);
    [mag,phase,wout] = bode(sys);
    %plot results, with frequency expressed at Hz
    subplot(2,1,1);
    semilogx(wout(:,1)/(2*pi), 20*log10(squeeze(mag)),'-'); zoom on; grid on;
%     title('Bode Plot')
    ylabel('Magnitude (dB)');
%     xline([1 3 142.6028 2*142.6028],'--',{'k_1 = 10 Nm','k_{37} = 86 Nm','höchste Frequenz','Nyquist-Frequenz'},'LabelVerticalAlignment','bottom')
%     xline([2 5],'r-',{'k_{min,SET}','k_{max,SET}'},'LabelVerticalAlignment','bottom','LabelHorizontalAlignment','center')
    xline([2 5],'r-')
    xline([2.46 5],'k--',{'k_{min,ACT} = 50 Nm','k_{max,ACT} = 220 Nm'},'LabelVerticalAlignment','bottom','LabelHorizontalAlignment','right')
%     xline([142.6028 2*142.6028],'--',{'höchste Frequenz','Nyquist-Frequenz'},'LabelVerticalAlignment','top')
%     xline([104/2/pi 896/2/pi],'--',{'Nst System 1','Nst System 37'},'LabelVerticalAlignment','bottom')
%     xline([104/2/pi 896/2/pi],'--')
%     xline([896/pi],'r-',{'Nyquist-Frequenz'})
    hold on
    subplot(2,1,2);
    semilogx(wout(:,1)/(2*pi), squeeze(phase),'-'); zoom on; grid on;
    xlabel('Frequency (Hz)'); ylabel('Phase (deg)');
%     xline([1 3 142.6028 2*142.6028],'--')
    xline([2 5],'r-')
    xline([2.46 5],'k--')
%     xline([104/2/pi 896/2/pi],'--')
%     xline([896/pi],'r-')
    hold on
end

% dt = 0.001;
% figure %zeitdiskretes System
% for i = 1:length(omega4_band_final)
% %     Am = sys_final_c{i,1}.A;
% %     Bm = sys_final_c{i,1}.B;
%     Am = Am_lin_global((i-1)*nxm+1:i*nxm,:);
%     Bm = Bm_lin_global((i-1)*nxm+1:i*nxm,:);
%     sys = c2d(ss(Am,Bm,[0 0 0 1],0),dt);
%     [mag,phase,wout] = bode(sys);
%     %plot results, with frequency expressed at Hz
%     subplot(2,1,1);
%     semilogx(wout(:,1)/(2*pi), 20*log10(squeeze(mag)),'-'); zoom on; grid on;
%     title('Amplitudengang'); xlabel('Frequenz (Hz)'); ylabel('Amplitudengang (dB)');
%     xline([1 3 142.6028 2*142.6028],'--',{'k_1 = 10 Nm','k_{37} = 86 Nm','höchste Frequenz','Nyquist-Frequenz'},'LabelVerticalAlignment','bottom')
%     hold on
%     subplot(2,1,2);
%     semilogx(wout(:,1)/(2*pi), squeeze(phase),'-'); zoom on; grid on;
%     title('Phasengang'); xlabel('Frequenz (Hz)'); ylabel('Phasengang (deg)');
%     xline([1 3 142.6028 2*142.6028],'--')
%     hold on
% end