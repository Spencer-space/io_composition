function [S,p] = IoCompBox(ha,hb,Te,c_bulk_desired,rA,output_opt)

p.N = 10000;
p.ha = ha;
p.hb = hb;
p.Te = Te;
p.c_bulk_desired = c_bulk_desired;

% Box model for Io including composition,
% ha = emplacement rate of component a
% hb = empalcement rate of component b
% c_bulk_desired = the desired bulk comp of Io, set negative if don't want to use
% rA does nothing if setting a desired bulk composition.

% Dimensional Parameters
p.K_0 = 10^-7; % reference permeability (m^2)
p.rho = 3000; % density (kg/m^3)
p.del_rho = 500; % density difference (kg/m^3)
p.g = 1.5; % gravity (m/s^2)
p.L = 4e5; % latent heat (J/kg)
p.kappa = 1e-6; % Thermal diffusivity (m^2/s)
p.c = 1200; % specific heat (J/kg/K)
p.T_l = 1350; % relative melting point (above surface T) (K)
p.n = 3; % permeability exponent
p.eta_l = 1; % basalt melt viscosity (Pas)
p.eta = 1e20; % viscosity (Pas)

p.r_s = 1820e3; % Io radius (m)
p.r_m = 700e3/1820e3; % normalise core radius

p.Psi_ref = 1e14/(4/3 * pi *(p.r_s^3 - 700e3^3)); % reference tidal heating (W/m^3)
p.q_0 = p.Psi_ref*p.r_s/p.rho/p.L; % reference velocity (m/s)
p.phi_0 = (p.q_0*p.eta_l/(p.K_0*p.del_rho*p.g))^(1/p.n); % reference porosity
p.P0 = p.eta/p.phi_0 *p.q_0/p.r_s;

p.opts = odeset('reltol',1e-6);

% Dimensionless Parameters
p.St = p.L/p.c/p.T_l; % Stefan number
p.Pe = p.q_0*p.r_s/p.kappa; % Peclet number
p.Psi = 1; % heating rate
p.T_A = 0.8; % non-dim melting point of component A
p.T_B = 1.0; % non-dim melting point of component B

p.comp_tolerance = 1e-3; % how close we want our result composition to be to the desired value

% boundary positions, empty for now
rc = 0;
rB = 0;

qp_rA = 0; % plumbing flux at top of mid-region. Needs predefining so it spans functions
qp_st = 0; % Stefan jump condition, predefine

% if a desired bulk composition is specified, do a newton iteration
if p.c_bulk_desired >= 0
    % Newton solve for the bulk composition
    % if no rA guess is specified, use this
    if rA == 0
        rA = (1 - (1-p.r_m^3)*p.c_bulk_desired)^(1/3); % initial guess for the position of r
        if rA>0.9
            rA=0.9;
        end
    end
    % if composition is an endmember, rA = rm
    if p.c_bulk_desired == 0 || p.c_bulk_desired == 1
        rA = p.r_m;
    end
    [S, c_bulk] = maincalc(); % get the bulk composition for the initial guess
    iter_comp = 0;
    while abs(c_bulk - p.c_bulk_desired) > p.comp_tolerance && iter_comp < 100 % doesn't really need to be that accurate
        r_Ahold = rA; % store the guess for r_AB before changing to get derivative
        % find numerical derivative of bulk wrt rA
        drA = 0.02;
        rA = r_Ahold + 0.01;
        if rA>1
            rA = 1;
        end
        try
            [Sp, bulk_p] = maincalc();
        catch
            warning('Position of rA too high, lowering')
            rA = r_Ahold;
            [Sp, bulk_p] = maincalc();
            drA = 0.01;
        end
        try
            rA = r_Ahold - 0.01;
            [Sm, bulk_m] = maincalc();
        catch
            warning('Position of rA too low, raising')
            rA = r_Ahold;
            [Sm, bulk_m] = maincalc();
            drA = 0.01;
        end
        dbulk_drab = (bulk_p - bulk_m)/drA;
        rA = r_Ahold; % restore back correct r_A
        rA = rA - (c_bulk - p.c_bulk_desired)/dbulk_drab;
        if rA>1
                rA = 1-1e-5;
            end
        try
            [S, c_bulk] = maincalc();
        catch
            % if fails here try accepting only a small part of newton step
            rA = rA + (c_bulk - p.c_bulk_desired)/dbulk_drab;
            rA = rA - 0.1*(c_bulk - p.c_bulk_desired)/dbulk_drab;
            try
                [S, c_bulk] = maincalc();
            catch
                error('Cannot find solution, effective crustal emplacement rate probably too low');
            end
        end
        iter_comp = iter_comp + 1;

        if iter_comp >= 100 && abs(c_bulk - p.c_bulk_desired) > p.comp_tolerance
            warning('Exceeded netwton iteration limit in search for rA')
        end
    end
    
else
    [S, c_bulk] = maincalc(); % get the bulk composition for the initial guess
end

if output_opt==1
    % dimensionalise
    S.T = S.T*p.T_l + 150;
    S.Tp = S.Tp*p.T_l + 150;
    S.q = S.q*p.q_0*100*60*60*24*365; % cm/yr
    S.qp = S.qp*p.q_0*100*60*60*24*365; % cm/yr
    S.u = S.u*p.q_0*100*60*60*24*365; % cm/yr
    S.M = S.M*p.q_0/p.r_s *60*60*24*365*1e6;
    S.phi = S.phi * p.phi_0;
    S.P = S.P * p.P0;
    S.r = S.r*p.r_s/1000;
    
    p.rA = rA;
    p.rB = rB;
    p.rc = rc;
    p.bulk_comp = c_bulk;
    
    IoCompBox_plotting(S,p); % plot up
elseif output_opt ==2
    % to read into petsc only need starting H and c.
    IN = zeros(2*p.N,1);
    c = smooth(S.c,100);
    H = smooth(S.T + p.St*p.phi_0*S.phi,100);
    qp = smooth(S.qp,100);
    Tp = smooth(S.Tp,100);
    cp = smooth(S.cp,100);
    for i=1:p.N
        IN(2*i-1) = H(i);
        IN(2*i) = c(i);
        INpipe(3*i-2) = qp(i);
        INpipe(3*i-1) = Tp(i);
        INpipe(3*i) = cp(i);
    end
    filename = sprintf('input_ha%03d_hb%03d',p.ha,p.hb);
    filenamepipe = sprintf('inputpipe_ha%03d_hb%03d',p.ha,p.hb);
    filename = fullfile('inputs',filename);
    filenamepipe = fullfile('inputs',filenamepipe);
    PetscBinaryWrite(filename,IN);
    PetscBinaryWrite(filenamepipe,INpipe);
else
    % dimensionalise
    S.T = S.T*p.T_l + 150;
    S.Tp = S.Tp*p.T_l + 150;
    S.q = S.q*p.q_0*100*60*60*24*365; % cm/yr
    S.qp = S.qp*p.q_0*100*60*60*24*365; % cm/yr
    S.u = S.u*p.q_0*100*60*60*24*365; % cm/yr
    S.M = S.M*p.q_0/p.r_s *60*60*24*365*1e6;
    S.phi = S.phi * p.phi_0;
    S.P = S.P * p.P0;
    S.r = S.r*p.r_s/1000;
    
    p.rA = rA;
    p.rB = rB;
    p.rc = rc;
    p.bulk_comp = c_bulk;
end

% ----------------------------------------------------------------------- %
% -------------------------Function Definitions-------------------------- %
% ----------------------------------------------------------------------- %
% maincalc - Calculate the full solution for a given position rA,     %
%            calculates bulk composition first, and proceeds to get whole %
%            solution if the ind tells us we're using the final rA    %

function [S, c_bulk] = maincalc()
    if p.c_bulk_desired ~= 0 && p.c_bulk_desired ~= 1
        % Use shooting method to find mid region thickness, see functions for details
        [rB,a,b,iter] = bisect(p.r_m,rA-1e-5,@(rB) shootrB(rB),1e-6); % shooting method to find thickness of region 2
        [x2,y2] = ode15s(@(x,y)odes1(x,y,p.T_B,p.hb),[rB rA],[p.Psi/3 *(rB^3 - p.r_m^3); p.T_B; 0],p.opts); % solve region 2 on the current assumed domain
        qp_rA = y2(end,1)/rA^2; % extract pipe flux at base
        qp_st = -1/(p.St*p.Pe) *y2(end,3)/rA^2; % instantaneous flux from Stefan jump condition
    else
        rB = p.r_m;
        qp_rA = 0;
        qp_st = 0;
        x2 = 0;
        y2 = [0,0,0];
    end
    
    % Now do a shooting method for the top region
    if p.c_bulk_desired ~= 0
        cp = 1; % initial guess of composition in crust is pure fusible (it's usually this)
    else
        cp = 0;
    end
%     rc_new = 1; % initially set base of crust out of way
%     rc_old = 0;
%     hhat_iter = 0;
    % this while loop removes the non-linearity of allowing hhat_crust to vary within bisect.
%     while abs(rc_new - rc_old) > 1e-6 && hhat_iter < 50
%         rc_old = rc_new;
%         p.hhat_crust = p.hb*(cp<=0.1) + p.ha*(cp>0.9) + (p.hb + (p.ha-p.hb)*(cp-0.1)/0.8)*(cp>0.1 & cp<0.9);
%         [rc,a,b,iter] = bisect(rA,1-1e-5,@(rc) shootrc(rc),1e-6); % shooting method to find thickness of region 2
%         qp_rc = max(0,qp_rA*rA^2/rc^2 - p.hb*(p.T_B-p.T_A)*(rc^3 - rA^3)/(3*rc^2));
%         q_rc = p.Psi/3 *(rc - rA^3/rc^2) + (1+(p.T_B-p.T_A)/p.St)*min(qp_rA*rA^2/rc^2,p.hb*(p.T_B-p.T_A)*(rc^3-rA^3)/(3*rc^2)) + qp_st*rA^2/rc^2;
%         Tp = qp_rc*p.T_B/(qp_rc+q_rc) + q_rc*(p.T_A*(p.c_bulk_desired~=0)+p.T_B*(p.c_bulk_desired==0))/(qp_rc+q_rc); % plumbing temp is weighted avg of material in plumbing system
%         cp = q_rc*(1*(p.c_bulk_desired~=0) + 0*(p.c_bulk_desired==0))/(q_rc + qp_rc);
%         rc_new = rc;
%         hhat_iter = hhat_iter + 1;
%     end
%     if hhat_iter == 50
%         warning('Max iterations over variable hhat')
%     end
    p.hhat_crust = p.ha;
    [rc,a,b,iter] = bisect(rA,1-1e-5,@(rc) shootrc(rc),1e-6); % shooting method to find thickness of region 2
    qp_rc = max(0,qp_rA*rA^2/rc^2 - p.hb*(p.T_B-p.T_A)*(rc^3 - rA^3)/(3*rc^2));
    q_rc = p.Psi/3 *(rc - rA^3/rc^2) + (1+(p.T_B-p.T_A)/p.St)*min(qp_rA*rA^2/rc^2,p.hb*(p.T_B-p.T_A)*(rc^3-rA^3)/(3*rc^2)) + qp_st*rA^2/rc^2;
    Tp = qp_rc*p.T_B/(qp_rc+q_rc) + q_rc*(p.T_A*(p.c_bulk_desired~=0)+p.T_B*(p.c_bulk_desired==0))/(qp_rc+q_rc); % plumbing temp is weighted avg of material in plumbing system
    cp = q_rc*(1*(p.c_bulk_desired~=0) + 0*(p.c_bulk_desired==0))/(q_rc + qp_rc);
    [x4,y4] = ode15s(@(x,y)odes1(x,y,Tp,p.hhat_crust),[rc 1],[(qp_rc+q_rc)*rc^2; p.T_A*(p.c_bulk_desired~=0) + p.T_B*(p.c_bulk_desired==0); 0],p.opts);
    % Get solutions
    r = linspace(p.r_m,1,p.N); % setup position vector

    S.T(r<rB) = p.T_B; % T in r1
    S.T(rB<=r & r<rA) = interp1(x2,y2(:,2),r(rB<=r & r<rA)); % interpolate T in r2
    S.T(rA<=r & r<=rc) = p.T_A*(p.c_bulk_desired~=0) + p.T_B*(p.c_bulk_desired==0); % T in r3
    S.T(r>rc) = interp1(x4,y4(:,2),r(r>rc)); % interpolate T in r4
    
    S.Tp(r<rB) = p.T_B; % T in r1
    S.Tp(rB<=r & r<rA) = p.T_B; % interpolate T in r2
    S.Tp(rA<=r & r<=rc) = p.T_B*(p.c_bulk_desired<1) + p.T_A*(p.c_bulk_desired==1); % T in r3
    S.Tp(r>rc) = Tp; % interpolate T in r4

    S.qp(r<rB) = 0*r(r<rB);
    S.qp(rB<=r & r<rA) = interp1(x2,y2(:,1)./x2.^2,r(rB<=r & r<rA)); % interpolate qp in r2
    S.qp(rA<=r & r<=rc) = max(0,qp_rA*rA^2./r(rA<=r & r<=rc).^2 - p.hb*(p.T_B-p.T_A)*(r(rA<=r & r<=rc).^3-rA^3)./(3*r(rA<=r & r<=rc).^2)); % qp in r3 (can't go negative)
    S.qp(r>rc) = interp1(x4,y4(:,1)./x4.^2,r(r>rc)); % interpolate qp in r4
    
    S.q(r<rB) = p.Psi/3 *(r(r<rB) - p.r_m^3./r(r<rB).^2); % q in r1
    S.q(rB<=r & r<rA) = 0*r(rB<=r & r<rA); % q in r2
    S.q(rA<=r & r<=rc) = p.Psi/3 *(r(rA<=r & r<=rc) - rA^3./r(rA<=r & r<=rc).^2) + (1+(p.T_B-p.T_A)/p.St)*min(qp_rA*rA^2./r(rA<=r & r<=rc).^2,p.hb*(p.T_B-p.T_A)*(r(rA<=r & r<=rc).^3-rA^3)./(3*r(rA<=r & r<=rc).^2)) + qp_st*rA^2./r(rA<=r & r<=rc).^2;
    S.q(r>rc) = 0*r(r>rc); % q in r4
    
    S.u = - S.qp - S.q; % u comes from continuity
    
    S.c(r<rB) = 0;
    S.c(rB<=r & r<rA) = 0;
    S.c(rA<=r & r<=rc) = -S.q(rA<=r & r<=rc)*(1*(p.c_bulk_desired~=0)+0*(p.c_bulk_desired==0))./S.u(rA<=r & r<=rc);
    S.c(r>rc) = cp *r(r>rc).^0;
    
    S.cp(r<rB) = 0;
    S.cp(rB<=r & r<rA) = 0;
    S.cp(rA<=r & r<=rc) = 0*(p.c_bulk_desired<1) + 1*(p.c_bulk_desired==1);
    S.cp(r>rc) = cp *r(r>rc).^0;
    
    S.phi = S.q.^(1/p.n); % phi from scaled Darcy q = phi^n
    S.M = (p.hb + (p.ha-p.hb)*S.cp).*(S.Tp - S.T) .*(S.qp>0).*(S.T>p.Te);
    S.P = -(p.St*p.Psi + S.M.*(S.Tp-S.T+p.St))./(p.St*S.phi);
    S.r = r;
    
    c_bulk = trapz(r,r.^2.*S.c)./trapz(r,r.^2);
    if p.c_bulk_desired == 0
        c_bulk = 0;
    elseif p.c_bulk_desired == 1
        c_bulk = 1;
    end
    
end
    % ----------------------------------------------------------------------- %
function [m,a,b,iter] = bisect(a,b,shoot,tol)
%     figure(1); clf;
    iter = 0;
    fa = shoot(a); fb = shoot(b);
    if fa*fb > 0
        warning('bisect: Signs at ends of interval are the same');
        % Try looking for another interval
        for tmp = 0.1:.1:1
            c = a+tmp*(b-a);
            fc = shoot(c); 
            if fa*fc < 0, b = c; fb = fc; warning(['bisect: Using reduced interval ',num2str([a b])]); break; end
        end
        % Otherwise give up
        if tmp==1
            warning('bisect: Could not find an interval with a sign change');
            m = NaN;
            return;
        end
    end
    while b-a>tol
        iter = iter+1;
        m = (a+b)/2;
        fm = shoot(m);
        if fm*fa<0
           b = m;
        else
           a = m;
           fa = fm;
        end
    end
end

% ----------------------------------------------------------------------- %

function out = shootrB(rB)
    % solve ODEs for a guess of the region 3 thickness, return T at top of region 3
    % Guesses for qp and dT take account of whether qp is zero or not. If it's zero in dT case, use q_e BC with q_e = 0.
    [x1,y1] = ode15s(@(x,y)odes1(x,y,p.T_B,p.hb),[rB rA],[p.Psi/3 *(rB^3 - p.r_m^3); p.T_B; 0],p.opts);
    out = y1(end,2) - p.T_A;
end

% ----------------------------------------------------------------------- %

function out = shootrc(rc)
    qp_rc = max(0,qp_rA*rA^2/rc^2 - p.hb*(p.T_B-p.T_A)*(rc^3 - rA^3)/(3*rc^2));
    q_rc = p.Psi/3 *(rc - rA^3/rc^2) + (1+(p.T_B-p.T_A)/p.St)*min(qp_rA*rA^2/rc^2,p.hb*(p.T_B-p.T_A)*(rc^3-rA^3)/(3*rc^2)) + qp_st*rA^2/rc^2;
    Tp = qp_rc*p.T_B/(qp_rc+q_rc) + q_rc*(p.T_A*(p.c_bulk_desired~=0)+p.T_B*(p.c_bulk_desired==0))/(qp_rc+q_rc); % plumbing temp is weighted avg of material in plumbing system
    if isnan(Tp)
        if p.c_bulk_desired==0
            Tp = 0;
        else
            Tp = 1;
        end
    end
    % solve ODEs for a guess of the region 3 thickness, return T at top of region 3
    % Guesses for qp and dT take account of whether qp is zero or not. If it's zero in dT case, use q_e BC with q_e = 0.
    [x1,y1] = ode15s(@(x,y)odes1(x,y,Tp,p.hhat_crust),[rc 1],[(qp_rc+q_rc)*rc^2; p.T_A*(p.c_bulk_desired~=0) + p.T_B*(p.c_bulk_desired==0); 0],p.opts);
    if max(x1) ~= 1
        disp(max(x1));
    end
    out = y1(end,2);
end

% ----------------------------------------------------------------------- %

function dydx = odes1(x,y,Tp,hhat)
    % stiff ODE solver for the 3 coupled 1st order equations
    qp = y(1,:)./x.^2; % y(1) = r^2*qp
    T = y(2,:);
    dT = y(3,:); % y(3) = r^2*dT/dr
    dydx(1,:) = - x.^2*hhat*(Tp-T).*(qp>0).*(T>p.Te); % M = 0 if qp<0
    dydx(2,:) = dT./x.^2;
    dydx(3,:) = -p.Pe*qp.*dT + x.^2.*( - p.Pe*p.St*p.Psi - p.Pe*hhat*(Tp-T)*(Tp-T + p.St).*(qp>0).*(T>Te)); % Energy equation rewritten for d2T/dr2. M = 0 if qp<0
end

% ----------------------------------------------------------------------- %

% ----------------------------------------------------------------------- %
% ----------------------------------------------------------------------- %

end