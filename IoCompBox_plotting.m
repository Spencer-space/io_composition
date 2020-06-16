function IoCompBox_plotting(S,p)

figure('Units','centimeters','Position',[0 0 70 30],'PaperPositionMode','auto');

%Temperature Plotting
fig1 = subplot(1,5,1);
line = plot(fig1,S.T,S.r,'LineWidth',1.7);
hold on
line = plot(fig1,S.Tp,S.r,'Linewidth',1.7);

ymin = 600;
ymax = 1820;

line = plot(fig1,[0 2100],[700 700],'-k'); %core line

xlim([0 1600])
ylim([ymin ymax])

set(fig1,'Units','normalized','YTick',600:200:1800,'FontUnits','points','FontSize',20,'FontName','Times')
ylabel(fig1,{'Radius  (km) '},'FontUnits','points', 'interpreter','latex','FontSize', 22,'FontName','Times')
xlabel(fig1,{'Temperature $ (K) $'},'FontUnits','points','interpreter','latex','FontSize', 22,'FontName','Times')
title(fig1,'Temperature Profile', 'FontUnits','points','Fontweight','normal','FontSize',24,'FontName','Times')

hold off

% Porosity Plotting
fig2 = subplot(1,5,2);
line = plot(fig2,S.phi*100,S.r,'LineWidth',1.7);
hold on
line = plot(fig2,[0 max(5,1.2*max(S.phi*100))],[700 700],'-k'); %core line

xlim([0 max(5,1.2*max(S.phi*100))])
ylim([ymin ymax])

set(fig2,'Units','normalized','YTick',600:200:1800,'FontUnits','points','FontSize',20,'FontName','Times')

xlabel(fig2,{'Porosity (\%)'},'FontUnits','points','interpreter','latex','FontSize', 22,'FontName','Times')
title(fig2,'Porosity Profile', 'FontUnits','points','Fontweight','normal','FontSize',24,'FontName','Times')
set(gca,'Yticklabel',[]) 
hold off

% Flux Plotting
fig3 = subplot(1,5,3);
line = plot(S.q,S.r,'linewidth',1.7);
hold on
line = plot(S.u,S.r,'LineWidth',1.7);
line = plot(S.qp,S.r,'linewidth',1.7);

line = plot(fig3,[-10 10],[700 700],'-k'); %core line

line = plot(fig3,zeros(2),linspace(ymin,ymax,2),'-k');

xlim([-10 10])
ylim([ymin ymax])

set(fig3,'Units','normalized','YTick',600:200:1800,'FontUnits','points','FontSize',20,'FontName','Times')

xlabel(fig3,{'Flux (cm/yr)'},'FontUnits','points','interpreter','latex','FontSize', 22,'FontName','Times')
legend(fig3,'Darcy','Solid','Plumbing','location','southeast')
set(gca,'Yticklabel',[])
title(fig3,'Flux Profiles', 'FontUnits','points','Fontweight','normal','FontSize',24,'FontName','Times')
hold off

% Composition Plotting
fig4 = subplot(1,5,4);
line = plot(S.c,S.r,'linewidth',1.7);
hold on
line = plot(S.cp,S.r,'linewidth',1.7);

line = plot(fig4,[0 1],[700 700],'-k'); %core line

xlim([0 1])
ylim([ymin ymax])

set(fig4,'Units','normalized','YTick',600:200:1800,'FontUnits','points','FontSize',20,'FontName','Times')
xlabel(fig4,{'Composition (frac A)'},'FontUnits','points','interpreter','latex','FontSize', 22,'FontName','Times')
legend(fig4,'Solid','Plumbing','location','southeast')
title(fig4,'Composition Profiles', 'FontUnits','points','Fontweight','normal','FontSize',24,'FontName','Times')
set(gca,'Yticklabel',[])

% Emplacement Plotting
fig5 = subplot(1,5,5);
line = plot(S.M,S.r,'linewidth',1.7);
hold on

line = plot(fig5,[0 max(1.2*max(S.M),1)],[700 700],'-k'); %core line

xlim([0 max(1.2*max(S.M),1)])
ylim([ymin ymax])

set(fig5,'Units','normalized','YTick',600:200:1800,'FontUnits','points','FontSize',20,'FontName','Times')
xlabel(fig5,{'Emplacement rate (s$^{-1}$)'},'FontUnits','points','interpreter','latex','FontSize', 22,'FontName','Times')
title(fig5,'Emplacement Profile', 'FontUnits','points','Fontweight','normal','FontSize',24,'FontName','Times')
set(gca,'Yticklabel',[])

linkaxes([fig1 fig2 fig3 fig3 fig4 fig5],'y');

% AxesHandle=findobj(gcf,'Type','axes');
% plot_width = (1-0.05-4*0.025-0.01)/4;
% for (i=1:7)
%     if i < 5
%         plot_ind = i;
%         set(AxesHandle(num_plots+1-i),'Position',[0.05+(plot_ind-1)*(plot_width+0.025),0.56,plot_width,0.42]);
%     else
%         plot_ind = i - 4;
%         set(AxesHandle(num_plots+1-i),'Position',[0.05+(plot_ind-1)*(plot_width+0.025),0.06,plot_width,0.42]);
%     end
% end