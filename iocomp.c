#include "iocomp.h"
#include <errno.h>
#include <sys/types.h>
#include <pwd.h>
#include <sys/stat.h>

/* Preamble - Setup
   X - Solution vector filled when solving for the next timestep
   R - Residual vector for timestepping
   Xold - Solution vector for the current time when solution is known
   
   Using a staggered grid, velocity is positioned between nodes, with the last velocity point in the vector being redundant
*/

PetscClassId classid[8];

/*-----------------------------------------------------------------------*/
int main(int argc,char **argv)
/*-----------------------------------------------------------------------*/
{
  AppCtx         user;
  PetscErrorCode ierr;
  
  /* always initialize PETSc first */
  PetscInitialize(&argc,&argv,(char *)0,PETSC_NULL);
  user.comm = PETSC_COMM_WORLD;
  
  /* set hardwired options */
  PetscOptionsSetValue(NULL,"-ksp_gmres_restart","300");
  //PetscOptionsSetValue(NULL,"-ksp_monitor","");
  //PetscOptionsSetValue(NULL,"-snes_monitor","");
  PetscOptionsSetValue(NULL,"-P_snes_linesearch_type","basic");
  PetscOptionsSetValue(NULL,"-H_snes_linesearch_type","basic");
  PetscOptionsSetValue(NULL,"-H_pc_type","lu");
  PetscOptionsSetValue(NULL,"-H_snes_stol","1.0e-8");
  PetscOptionsSetValue(NULL,"-H_snes_rtol","1.0e-6");
  PetscOptionsSetValue(NULL,"-H_snes_abstol","1.0e-7");
  PetscOptionsSetValue(NULL,"-snes_converged_reason","");
  PetscOptionsInsert(NULL,&argc,&argv,PETSC_NULL);
  
  /* set up parameter structure */
  ierr = PetscBagCreate(user.comm,sizeof(Parameter),&(user.bag));  CHKERRQ(ierr);
  ierr = PetscBagGetData(user.bag,(void**)&user.param);  CHKERRQ(ierr);
  ierr = PetscBagSetName(user.bag,"par","parameters for compaction problem");  CHKERRQ(ierr);
  ierr = ParameterSetup(user);CHKERRQ(ierr);
  
  /* Create output directory if it doesn't already exist */
  ierr = projCreateDirectory("outputs"); CHKERRQ(ierr);
  ierr = projCreateDirectory("SS_outputs"); CHKERRQ(ierr);
  /* Set up solution vectors and SNES */
  ierr = DMDASNESsetup(&user);CHKERRQ(ierr);
  
  /* if a restart file is specified, load it in, else set-up with initialisation */
  if (user.param->restart_step != 0) {
    ierr = RestartFromFile(&user);CHKERRQ(ierr);
  } else {
    /* Initialise vectors and parameters that need calculating */
    ierr = Initialisation(&user,user.XP,user.XH,user.XPipe,user.Haux,user.Paux);CHKERRQ(ierr);
  }

  /* Main timestepping function */
  ierr = TimeStepping(&user);CHKERRQ(ierr);
  
  /* Clean-up by deleting vectors etc */
  ierr = CleanUp(&user);CHKERRQ(ierr);
  
  return 0;
}

/* ------------------------------------------------------------------- */
PetscErrorCode Initialisation(AppCtx *user, Vec XP, Vec XH, Vec XPipe, Vec Haux, Vec Paux)
/* ------------------------------------------------------------------- */
{
  Parameter       *p = user->param;
  PetscErrorCode  ierr;
  FieldP          *xp;
  FieldH          *xh;
  FieldPipe       *xpipe;
  AuxFieldP       *paux;
  AuxFieldH       *haux;
  PetscInt        i;
  PetscReal       dr;
  PetscFunctionBegin;
  
  /* set non-dim base */
  p->base = p->R_cmb/p->R;
  
  /* Spatial grid spacing (constant) */
  dr = (1-p->base)/(p->ni-2);
  
  /* Reference values and non-dim values */
  p->psi_0 = p->Psi_0/(4.0/3.0 * PETSC_PI * (pow(p->R,3)-pow(p->R_cmb,3))); // set reference constant heating rate in W/m3
  p->Pe = p->psi_0*p->R*p->R/(p->L*p->kappa*p->rho_0); // Peclet number
  p->Pec = p->psi_0*p->R*p->R/(p->L*p->D*p->rho_0); // compositional Peclet number
  p->St = p->L/(p->ce*p->T_0); // Stefan number
  p->phi_0 = pow(p->psi_0*p->R*p->eta_l/(p->L*p->rho_0*p->K_0*p->del_rho*p->g),1/p->perm); // reference porosity
  p->zeta_0 = p->eta/p->phi_0;
  p->delta = p->zeta_0*p->K_0*pow(p->phi_0,p->perm)/(p->eta_l*p->R*p->R);
  p->P_0 = p->zeta_0*p->psi_0/(p->rho_0*p->L); // reference pressure
  
  /* Get the initial (empty) solution vectors */
  ierr = DMDAVecGetArray(user->daP,XP,&xp);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(user->daH,XH,&xh);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(user->daPipe,XPipe,&xpipe);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(user->daPaux,Paux,&paux);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(user->daHaux,Haux,&haux);CHKERRQ(ierr);

  /* if not initialising from some other state, set up one here */
  for (i=0; i<p->ni; i++) {
    paux[i].r  = p->base + dr*(i-0.5); // set up position vector (cell centers)
    xh[i].c = p->bulk_comp;
    xh[i].H = p->T_B + (p->T_A-p->T_B)*(1.0-exp(-xh[i].c/p->gamma))/(1.0-exp(-1.0/p->gamma));
  }
  /* If one-component the other emplacement constant is unused, so set it to zero for neat outputting */
  if (p->bulk_comp == 1) {
    p->hhat_B = 0;
  } else if (p->bulk_comp == 0) {
    p->hhat_A = 0;
  }
  
  ierr = PAuxParamsCalc(user,xp,xpipe,haux,paux);CHKERRQ(ierr);
  
  ierr = DMDAVecRestoreArray(user->daP,XP,&xp);CHKERRQ(ierr); // restore vector
  ierr = DMDAVecRestoreArray(user->daH,XH,&xh);CHKERRQ(ierr); // restore vector
  ierr = DMDAVecRestoreArray(user->daPipe,XPipe,&xpipe);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(user->daPaux,Paux,&paux);CHKERRQ(ierr); // restore vector
  ierr = DMDAVecRestoreArray(user->daHaux,Haux,&haux);CHKERRQ(ierr); // restore vector
  
  ierr = VecCopy(user->XP,user->XPtheta);CHKERRQ(ierr); // P theta vector initially takes the setup state
  ierr = VecCopy(user->Paux,user->Pauxtheta);CHKERRQ(ierr); // P theta vector initially takes the setup state
  
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
PetscErrorCode TimeStepping(AppCtx *user)
/* ------------------------------------------------------------------- */
{
  Parameter    *p = user->param;
  PetscBool   STEADY=PETSC_FALSE, converged;
  PetscErrorCode  ierr;
  PetscReal   dt = p->dt;
  PetscReal   Pc_original;
  PetscReal   eps=1e-7, RH2, RP2, RPipe2;
  PetscInt    SNESit;
  PetscFunctionBegin;
  
  /* Output initial state */
  ierr = DoOutput(user,p->it,STEADY);CHKERRQ(ierr);
  
  /* If big pc used, set it small to be gradually increased */
  Pc_original = p->Pc;
  if (p->Pc > 0.5) {
    p->Pc = 0.5;
  }
  
  while (p->t <= p->tmax && p->it <= p->ns) {
    printf("########## step %d ################ initial timestep %+1.4e \n",p->it,p->dt);
    ierr = VecCopy(user->XP,user->XPold);CHKERRQ(ierr); // copy current timestep's solution and stores it as Xold.
    ierr = VecCopy(user->XH,user->XHold);CHKERRQ(ierr); // copy current timestep's solution and stores it as Xold.
    ierr = VecCopy(user->XPipe,user->XPipeold);CHKERRQ(ierr); // copy current timestep's solution and stores it as Xold.
    ierr = VecCopy(user->Haux,user->Hauxold);CHKERRQ(ierr);
    ierr = VecCopy(user->Paux,user->Pauxold);CHKERRQ(ierr);
    
    converged = PETSC_FALSE;
    FLAGH:
    printf("   >>>>>>>>>>>>>>>>>>>>[step %d] timestep %+1.4e \n",p->it,p->dt);
    if (p->dt < 1e-12 && p->theta == 1) {
      ierr = PetscPrintf(user->comm,"TRYING AN EXPLICIT TIMESTEP \n");CHKERRQ(ierr);
      // try making it explicit
      p->theta = 0;
      p->dt = 1e-4;
    } else if (p->dt < 1e-12) {
      ierr = CHANGE_CONSOLE_COLOUR(user->comm,BOLD_RED);CHKERRQ(ierr);
      ierr = PetscPrintf(user->comm,"TIMESTEP REDUCED BELOW TOLERANCE at it %i \n",p->it);CHKERRQ(ierr);
      ierr = CHANGE_CONSOLE_COLOUR(user->comm,COLOUR_RESET);CHKERRQ(ierr);
    }
     
    for(SNESit=0; SNESit<p->SNESit; SNESit++) {
      
      p->true_solve_flag = 0; // flag used in debugging, identify when you're in the true solve
      /* Do enthalpy and composition solve */
      ierr = SNESSolve(user->snesH,PETSC_NULL,user->XH);CHKERRQ(ierr); // non-linear solver
      ierr = SNESGetConvergedReason(user->snesH,&user->reasonH);CHKERRQ(ierr);
      ierr = SNESComputeFunction(user->snesH,user->XH,user->RH);CHKERRQ(ierr); // make sure I have the correct auxilliary parameters
      
      /* Now do pressure solve */
      ierr = SNESSolve(user->snesP,PETSC_NULL,user->XP);CHKERRQ(ierr); // non-linear solver
      ierr = SNESGetConvergedReason(user->snesP,&user->reasonP);CHKERRQ(ierr);
      ierr = SNESComputeFunction(user->snesP,user->XP,user->RP);CHKERRQ(ierr); // make sure I have the correct auxilliary parameters
      
      /* Now do pipe solve */
      ierr = SNESSolve(user->snesPipe,PETSC_NULL,user->XPipe);CHKERRQ(ierr); // non-linear solver
      ierr = SNESGetConvergedReason(user->snesPipe,&user->reasonPipe);CHKERRQ(ierr);
      ierr = SNESComputeFunction(user->snesPipe,user->XPipe,user->RPipe);CHKERRQ(ierr); // make sure I have the correct auxilliary parameters
      
      /* Pseudo-timestep if pipe solver failed */
      if (user->reasonPipe < 0) {
        printf("PIPE SOLVER FAILED reason %d - Entering pseudo timestepper \n",user->reasonPipe);
        ierr = PseudoTransient(user,eps); // do PseudoTransient stepping
        ierr = SNESComputeFunction(user->snesPipe,user->XPipe,user->RPipe);CHKERRQ(ierr); // reevaluate main pipe function
        ierr = VecNorm(user->RPipe,NORM_2,&RPipe2);CHKERRQ(ierr);
        if (RPipe2 < eps) {
          printf("PIPE PSEUDO TIMESTEPPER SUCCEEDED - continuing \n");
          user->reasonPipe = user->reasonPipePT; // if pseudo-stepper worked, this will allow the code to continue
        } else {
          printf("PIPE PSEUDO TIMESTEPPER FAILED - timestep will be reduced \n");
        }
      }
      
       /* Print out the Jacobian if it seems to be broken (look for NaNs, 0's on main diagonal, infs etc). */
       /* change the output viewer to a binary if want to open in matlab for a look */
//      if (p->it > 1) {
//        if (user->reasonH<0 || user->reasonP<0 || user->reasonPipe<0) {
//         {
//           Mat J;
//
//           PetscPrintf(user->comm,"Hr = %d, Pr = %d, Pipr = %d \n",user->reasonH,user->reasonP,user->reasonPipe);
//           ierr = SNESGetJacobian(user->snesPipe,&J,NULL,NULL,NULL);CHKERRQ(ierr);
//           ierr = SNESComputeJacobian(user->snesPipe,user->XPipe,J,J);CHKERRQ(ierr);
//           MatView(J,PETSC_VIEWER_STDOUT_WORLD);
//           ierr = DoOutput(user,p->it,STEADY); CHKERRQ(ierr);
//         }
//         exit(0);
//        }
//      }

      p->true_solve_flag = 1; // flag used in debugging, identify when you're in the true solve
      ierr = SNESComputeFunction(user->snesH,user->XH,user->RH);CHKERRQ(ierr);
      ierr = SNESComputeFunction(user->snesP,user->XP,user->RP);CHKERRQ(ierr);
      ierr = SNESComputeFunction(user->snesPipe,user->XPipe,user->RPipe);CHKERRQ(ierr);
      ierr = VecNorm(user->RH,NORM_2,&RH2);CHKERRQ(ierr);
      ierr = VecNorm(user->RP,NORM_2,&RP2);CHKERRQ(ierr);
      ierr = VecNorm(user->RPipe,NORM_2,&RPipe2);CHKERRQ(ierr);
      
      printf("[SNESit %d] reasons %d %d %d : norms %+1.4e  %+1.4e  %+1.4e\n",SNESit,user->reasonH,user->reasonP,user->reasonPipe,RH2,RP2,RPipe2);
      
      /*
       <DAM>
       Reduce dt less aggresively, 2x instead of 10x
      */
      if (RH2 < eps && RP2 < eps && RPipe2 < eps) {
        converged = PETSC_TRUE;
        break;
      }
      if (user->reasonH < 0) { p->dt = p->dt/2.0; goto FLAGH; } // if diverged from max-iterations, go back and try with smaller timestep
      if (user->reasonP < 0) { p->dt = p->dt/2.0; goto FLAGH; } // if diverged from max-iterations, go back and try with smaller timestep
      if (user->reasonPipe < 0) { p->dt = p->dt/2.0; goto FLAGH; } // if diverged from max-iterations, go back and try with smaller timestep
      /* if timestep is too small, exit, it's not doing anything */
      if (p->dt < 1e-12) {
        break;
      }
    }
    if (!converged && p->dt > 1e-12) {
      p->dt = p->dt/2.0; goto FLAGH;
    } else if (!converged) {
      ierr = CHANGE_CONSOLE_COLOUR(user->comm,BOLD_RED);CHKERRQ(ierr);
      ierr = PetscPrintf(user->comm,"TIMESTEP REDUCED BELOW TOLERANCE at it %i \n",p->it);CHKERRQ(ierr);
      ierr = CHANGE_CONSOLE_COLOUR(user->comm,COLOUR_RESET);CHKERRQ(ierr);
      goto FLAGERR;
    }
    
    p->t = p->t + p->dt; // update time
    p->theta = 1; // make implicit again if it was changed
    
    /*  Additional files to check outputs through run  */
    if ((p->out_freq != 0) && (p->it % p->out_freq == 0)) { ierr = DoOutput(user,p->it,STEADY); CHKERRQ(ierr); }
    if (p->it % 10 == 0 || p->dt < dt) {
      ierr = PetscPrintf(user->comm,"---------- Timestep - %i, time - %e -------- dt = %e ---\n",p->it,p->t,p->dt);CHKERRQ(ierr);
    }
    
    /* Every 100 timesteps check if steady state has been achieved, if so break */
    if(p->it % 100 == 0) {
      STEADY = SteadyStateChecker(user);
      if (STEADY && p->Pc == Pc_original) {
        ierr = CHANGE_CONSOLE_COLOUR(user->comm,BOLD_GREEN);CHKERRQ(ierr);
        ierr = PetscPrintf(user->comm,"STEADY STATE ACHIEVED AFTER %i TIMESTEPS \n",p->it);CHKERRQ(ierr);
        ierr = CHANGE_CONSOLE_COLOUR(user->comm,COLOUR_RESET);CHKERRQ(ierr);
        break;
      }
      /* if not using full desired Pc, increase */
      if (STEADY && p->Pc != Pc_original) {
        p->Pc = p->Pc + 0.2;
      }
      /* check that Pc hasn't been overshot */
      if (p->Pc > Pc_original) {
        p->Pc = Pc_original;
      }
    }
    p->it++;
    

    /*
     <DAM>
     Gradually increase dt (5%), but only if step is a multiple of 10.
     The value of 10 is arbitrary, we just want a rule to slowly grow dt.
     Always keep dt less than the user specified time step.
     We could get rid of p->it%10 and instead simply grow dt by a factor 1.005.
     That might be nicer and easier to understand.
    */
    if (p->it%10 == 0) {
      p->dt = 1.05 * p->dt;
    }
    
  }
  
  FLAGERR:ierr = DoOutput(user,p->it,STEADY);CHKERRQ(ierr);
  ierr = PetscPrintf(user->comm,"Computation stopped after %d timesteps, at time %e \n",p->it,p->t);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
PetscErrorCode PseudoTransient(AppCtx *user, PetscReal eps)
/* ------------------------------------------------------------------- */
{
  Parameter       *p = user->param;
  PetscErrorCode  ierr;
  Vec             XPipe_sum;
  PetscReal       dtauinv = 1.0/p->dtau;
  PetscInt        iter = 0;
  PetscReal       Pipe_L2 =1,Pipe_range[2];
  
  Vec             XPipePT, RPipePT, XPipeoldPT;
  SNES            snesPipePT;
  DM              daPipe = user->daPipe;
  Mat             J;
  const PetscInt  maxit = 100000;
  PetscReal       dtau_init = p->dtau; /* <DAM> save value - we will adjust it dynamically */
  
  /* Set up pseudo-transient vector */
  ierr = DMCreateGlobalVector(daPipe,&XPipePT); CHKERRQ(ierr);
  ierr = VecDuplicate(XPipePT,&RPipePT); CHKERRQ(ierr);
  ierr = VecDuplicate(XPipePT,&XPipeoldPT); CHKERRQ(ierr);
  
  /* set up nonlinear solver context for pipe */
  ierr = SNESCreate(user->comm,&snesPipePT);CHKERRQ(ierr);
  ierr = SNESSetOptionsPrefix(snesPipePT,"pipePT_");CHKERRQ(ierr);
  ierr = SNESSetFunction(snesPipePT,RPipePT,FormResidualPipePT,user);CHKERRQ(ierr);
    
  ierr = DMCreateMatrix(user->daPipe,&J);CHKERRQ(ierr);
  ierr = SNESSetJacobian(snesPipePT,J,J,SNESComputeJacobianDefaultColor,NULL);CHKERRQ(ierr);
  
  ierr = PetscObjectCompose((PetscObject)snesPipePT,"_SNES_XPipeoldPT_Vec_",(PetscObject)XPipeoldPT);CHKERRQ(ierr);
  PetscOptionsSetValue(NULL,"-pipePT_pc_type","lu");
  ierr = SNESSetFromOptions(snesPipePT);CHKERRQ(ierr);
  
  user->reasonPipePT = 0; // set out of the way
  
  ierr = VecDuplicate(user->XPipe,&XPipe_sum); CHKERRQ(ierr);
  ierr = VecCopy(user->XPipeold,XPipePT); CHKERRQ(ierr);
  
  /* loop while the norm of the difference between two timesteps is greater than some value, have a limit of max iterations */
  while (Pipe_L2 > eps && iter <= maxit) {
    ierr = VecCopy(XPipePT,XPipeoldPT); CHKERRQ(ierr); // copy the solution into the previous timestep solution
    ierr = SNESSolve(snesPipePT,NULL,XPipePT);CHKERRQ(ierr); // non-linear solver for mini-timestepper
    ierr = SNESGetConvergedReason(snesPipePT,&user->reasonPipePT);CHKERRQ(ierr);
    if (user->reasonPipePT < 0) break;
    
    ierr = SNESComputeFunction(snesPipePT,XPipePT,RPipePT);CHKERRQ(ierr); // make sure I have the correct auxilliary parameters
 
    VecCopy(XPipePT,XPipe_sum);
    VecAXPY(XPipe_sum,-1.0,XPipeoldPT);
    VecScale(XPipe_sum,dtauinv);
    VecAbs(XPipe_sum);
    VecMin(XPipe_sum,NULL,&Pipe_range[0]);
    VecMax(XPipe_sum,NULL,&Pipe_range[1]);
    printf("  PT %.4d : min,max(|X_i - Xold_i|) %+1.12e , %+1.12e\n",iter,Pipe_range[0],Pipe_range[1]);
    
    ierr = VecCopy(XPipePT,user->XPipe);CHKERRQ(ierr);
    ierr = SNESComputeFunction(user->snesPipe,user->XPipe,user->RPipe);CHKERRQ(ierr); // re-evaluate main pipe function
    VecNorm(user->RPipe,NORM_2,&Pipe_L2);
    printf("  PT %.4d : l2(target) %+1.12e\n",iter,Pipe_L2);
    
    /* grow dtau inversely proportional to steady-state residual */
    /* Note - Stuff defined inside these {} is only visible within {}, that is its 'scope' */
    {
      PetscReal dtaup;
      
      dtaup = p->dtau * (1.0/Pipe_range[1])*1.0e-2;
      if (dtaup < p->dtau) { dtaup = p->dtau; }
      
      printf("  PT %.4d : dt %+1.4e  (current) : dt_inv %+1.4e (scheduled)\n",iter,p->dtau,dtaup);
      p->dtau = dtaup;
      dtauinv = 1.0/p->dtau;
    }
    
    iter++;
  }
  
  p->dtau = dtau_init; /* reset this parameter for next PT solve */
  
  if (user->reasonPipePT < 0) {
    printf("MINI PIPE ITERATION NOT CONVERGED, continuing to reduce timestep \n");
  } else if (iter == maxit) {
    printf("MINI PIPE ITERATION DID NOT HIT STEADY STATE, L2 = %.e \n",Pipe_L2);
  } else {
    printf("MINI PIPE ITERATION CONVERGED\n");
  }
  ierr = VecCopy(XPipePT,user->XPipe); CHKERRQ(ierr); // put the transient solution into the main solution vector
  
  /* cleanup */
  ierr = PetscObjectCompose((PetscObject)snesPipePT,"_SNES_XPipeoldPT_Vec_",NULL);CHKERRQ(ierr);

  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = VecDestroy(&XPipePT);CHKERRQ(ierr);
  ierr = VecDestroy(&RPipePT);CHKERRQ(ierr);
  ierr = VecDestroy(&XPipeoldPT);CHKERRQ(ierr);
  ierr = SNESDestroy(&snesPipePT);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
PetscErrorCode FormResidualH(SNES snesH, Vec XH, Vec RH, void *ptr)
/* ------------------------------------------------------------------- */
{
  AppCtx          *user = (AppCtx*)ptr;
  Parameter       *p = user->param;
  PetscErrorCode  ierr;
  PetscInt        i, is, ie;
  const FieldH    *xh;
  FieldH          *rh, *xhold, *xhtheta;
  FieldP          *xptheta;
  FieldPipe       *xpipetheta;
  AuxFieldH       *haux, *hauxtheta;
  AuxFieldP       *pauxtheta;
  PetscFunctionBegin;
  
  /* get the residual array */
  ierr = DMDAVecGetArray(user->daH,RH,&rh);CHKERRQ(ierr);
  
  /* get the current guess of solution */
  ierr = DMDAVecGetArrayRead(user->daH,XH,&xh);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(user->daHaux,user->Haux,&haux);CHKERRQ(ierr);
  
  /* calculate auxilliary parameters associated with this solver */
  ierr = HAuxParamsCalc(user,xh,haux);CHKERRQ(ierr);
  
  /* theta method for XH */
  ierr = DMDAVecRestoreArrayRead(user->daH,XH,&xh);CHKERRQ(ierr);
  ierr = VecCopy(XH,user->XHtheta);CHKERRQ(ierr);
  ierr = VecAXPBY(user->XHtheta,1.0-p->theta,p->theta,user->XHold);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(user->daH,XH,&xh);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(user->daH,user->XHtheta,&xhtheta);CHKERRQ(ierr);
  /* theta method for Haux*/
  ierr = DMDAVecRestoreArray(user->daHaux,user->Haux,&haux);CHKERRQ(ierr);
  ierr = VecCopy(user->Haux,user->Hauxtheta);CHKERRQ(ierr);
  ierr = VecAXPBY(user->Hauxtheta,1.0-p->theta,p->theta,user->Hauxold);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(user->daHaux,user->Hauxtheta,&hauxtheta);CHKERRQ(ierr);
  /* theta method for XP */
  ierr = VecCopy(user->XP,user->XPtheta);CHKERRQ(ierr);
  ierr = VecAXPBY(user->XPtheta,1.0-p->theta,p->theta,user->XPold);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(user->daP,user->XPtheta,&xptheta);CHKERRQ(ierr);
  /* theta method for Paux */
  ierr = VecCopy(user->Paux,user->Pauxtheta);CHKERRQ(ierr);
  ierr = VecAXPBY(user->Pauxtheta,1.0-p->theta,p->theta,user->Pauxold);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(user->daPaux,user->Pauxtheta,&pauxtheta);CHKERRQ(ierr);
  /* theta method for pipe */
  ierr = VecCopy(user->XPipe,user->XPipetheta);CHKERRQ(ierr);
  ierr = VecAXPBY(user->XPipetheta,1.0-p->theta,p->theta,user->XPipeold);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(user->daPipe,user->XPipetheta,&xpipetheta);CHKERRQ(ierr);
  
  /* get the solution from last timestep */
  ierr = DMDAVecGetArray(user->daH,user->XHold,&xhold);CHKERRQ(ierr);
  
  /* get dimensions of LOCAL piece of grid (in parallel, this is not full grid) */
  ierr = DMDAGetGridInfo(user->daH,&is,0,0,&ie,0,0,0,0,0,0);CHKERRQ(ierr);
  
  /* apply boundary condition at r=0 */
  if (is==0) {
     i = 0; is++;
     rh[i].H  = xh[i].H - xh[i+1].H;
     rh[i].c  = xh[i].c - xh[i+1].c;
  }
  
  /* apply boundary condition at r=r_max */
  if (ie==p->ni) {
    i = p->ni-1; ie--;
    rh[i].H  = 0.5*(xh[i].H + xh[i-1].H) - p->H_end;
    if (xpipetheta[i].qp > 0) {
      rh[i].c  = 0.5*(xh[i].c + xh[i-1].c) - 0.5*(xpipetheta[i].cp + xpipetheta[i-1].cp); // surface is the erupted composition if erupting
    } else {
      rh[i].c = xh[i].c - xhold[i].c; // if not erupting, unchanging
    }
  }
  
  /* interior of the domain */
  for (i=is; i<ie; i++) {
    rh[i].H  = EnthalpyResidual(user,xh,xhold,xhtheta,xptheta,xpipetheta,hauxtheta,pauxtheta,i);
    if (p->bulk_comp < 1 && p->bulk_comp > 0) {
      rh[i].c = CompositionResidual(user,xh,xhold,xhtheta,xptheta,xpipetheta,hauxtheta,pauxtheta,i);
    } else {
      rh[i].c = xh[i].c - p->bulk_comp;
    }
  }
  
  /* clean up */
  ierr = DMDAVecRestoreArrayRead(user->daH,XH,&xh);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(user->daH,user->XHold,&xhold);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(user->daH,user->XHtheta,&xhtheta);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(user->daP,user->XPtheta,&xptheta);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(user->daPipe,user->XPipetheta,&xpipetheta);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(user->daHaux,user->Hauxtheta,&hauxtheta);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(user->daPaux,user->Pauxtheta,&pauxtheta);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(user->daH,RH,&rh);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
PetscErrorCode FormResidualP(SNES snesP, Vec XP, Vec RP, void *ptr)
/* ------------------------------------------------------------------- */
{
  AppCtx          *user = (AppCtx*)ptr;
  Parameter       *p = user->param;
  PetscReal       dr = (1 - p->base)/(p->ni-2);
  PetscErrorCode  ierr;
  PetscInt        i, is, ie;
  const FieldP    *xp;
  FieldP          *rp;
  FieldPipe       *xpipe;
  AuxFieldH       *haux;
  AuxFieldP       *paux;
  PetscFunctionBegin;
  
  /* get the residual array */
  ierr = DMDAVecGetArray(user->daP,RP,&rp);CHKERRQ(ierr);
  
  /* get the current guess of solution */
  ierr = DMDAVecGetArrayRead(user->daP,XP,&xp);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(user->daPaux,user->Paux,&paux);CHKERRQ(ierr);
  
  /* get vector of solution from other SNESs */
  ierr = DMDAVecGetArray(user->daPipe,user->XPipe,&xpipe);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(user->daHaux,user->Haux,&haux);CHKERRQ(ierr);
  
  /* get dimensions of LOCAL piece of grid (in parallel, this is not full grid) */
  ierr = DMDAGetGridInfo(user->daP,&is,0,0,&ie,0,0,0,0,0,0);CHKERRQ(ierr);
  
  /* calculate auxilliary parameters each Newton step from solution guess */
  ierr = PAuxParamsCalc(user,xp,xpipe,haux,paux);CHKERRQ(ierr);
  
  /* apply boundary condition at r=0 */
  if (is==0) {
     i = 0; is++;
     if (0.5*(haux[i].phi+haux[i+1].phi) > 1e-8) {
       rp[i].P  = xp[i].P - xp[i+1].P + dr*(1-p->phi_0*0.5*(haux[i].phi+haux[i+1].phi))/p->delta; // from q = 0 at base
     } else {
       rp[i].P = 0.5*(xp[i].P + xp[i+1].P);
     }
  }
  
  /* apply boundary condition at r=r_max */
  if (ie==p->ni) {
     i = p->ni-1; ie--;
     rp[i].P  = 0.5*(xp[i].P + xp[i-1].P); // no liquid overpressure at surface
  }
  
  /* interior of the domain */
  for (i=is; i<p->ni-1; i++) {
    rp[i].P  = PressureResidual(user,xp,xpipe,haux,paux,i);
  }
  
  /* clean up */
  ierr = DMDAVecRestoreArrayRead(user->daP,XP,&xp);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(user->daPipe,user->XPipe,&xpipe);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(user->daHaux,user->Haux,&haux);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(user->daPaux,user->Paux,&paux);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(user->daP,RP,&rp);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
PetscErrorCode FormResidualPipe(SNES snesPipe, Vec XPipe, Vec RPipe, void *ptr)
/* ------------------------------------------------------------------- */
{
  AppCtx          *user = (AppCtx*)ptr;
  Parameter       *p = user->param;
  PetscErrorCode  ierr;
  PetscInt        i, is, ie;
  const FieldPipe *xpipe;
  FieldPipe       *rpipe;
  FieldP          *xp;
  FieldH          *xh;
  AuxFieldH       *haux;
  AuxFieldP       *paux;
  PetscFunctionBegin;
  
  /* get the residual array */
  ierr = DMDAVecGetArray(user->daPipe,RPipe,&rpipe);CHKERRQ(ierr);
  
  /* get the current guess of solution */
  ierr = DMDAVecGetArrayRead(user->daPipe,XPipe,&xpipe);CHKERRQ(ierr);
  
  ierr = DMDAVecGetArray(user->daP,user->XP,&xp);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(user->daH,user->XH,&xh);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(user->daPaux,user->Paux,&paux);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(user->daHaux,user->Haux,&haux);CHKERRQ(ierr);
  
  /* get dimensions of LOCAL piece of grid (in parallel, this is not full grid) */
  ierr = DMDAGetGridInfo(user->daP,&is,0,0,&ie,0,0,0,0,0,0);CHKERRQ(ierr);
  
  /* apply boundary condition at r=0 */
  if (is==0) {
    i = 0; is++;
    rpipe[i].qp = xpipe[i].qp;
    rpipe[i].Tp = xpipe[i].Tp - haux[i].T;
    rpipe[i].cp = xpipe[i].cp - haux[i].cl;
  }
  
  /* apply boundary condition at r=r_max */
  if (ie==p->ni) {
     i = p->ni-1; ie--;
     rpipe[i].qp = xpipe[i].qp - xpipe[i-1].qp;
     rpipe[i].Tp = xpipe[i].Tp - xpipe[i-1].Tp;
     rpipe[i].cp = xpipe[i].cp - xpipe[i-1].cp;
  }
  
  /* interior of the domain */
  for (i=is; i<p->ni-1; i++) {
    rpipe[i].qp = qpResidual(user,xpipe,xp,haux,paux,i);
    if (p->bulk_comp > 0 && p->bulk_comp < 1) {
      rpipe[i].Tp = TpResidual(user,xpipe,xp,haux,paux,i);
      rpipe[i].cp = cpResidual(user,xpipe,xp,haux,paux,i);
    } else {
      rpipe[i].Tp = xpipe[i].Tp - p->T_A*(p->bulk_comp==1) - p->T_B*(p->bulk_comp==0);
      rpipe[i].cp = xpipe[i].cp - p->bulk_comp;
    }
  }
  
  /* clean up */
  ierr = DMDAVecRestoreArrayRead(user->daPipe,XPipe,&xpipe);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(user->daP,user->XP,&xp);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(user->daHaux,user->Haux,&haux);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(user->daPaux,user->Paux,&paux);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(user->daPipe,RPipe,&rpipe);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
PetscErrorCode FormResidualPipePT(SNES snesPipe, Vec XPipePT, Vec RPipePT, void *ptr)
/* ------------------------------------------------------------------- */
{
  AppCtx          *user = (AppCtx*)ptr;
  Parameter       *p = user->param;
  PetscErrorCode  ierr;
  PetscInt        i, is, ie;
  const FieldPipe *xpipe;
  FieldPipe       *rpipe, *xpipeoldPT;
  FieldP          *xp;
  AuxFieldH       *haux;
  AuxFieldP       *paux;
  Vec             XPipeoldPT = NULL;
  PetscFunctionBegin;

  ierr = PetscObjectQuery((PetscObject)snesPipe,"_SNES_XPipeoldPT_Vec_",(PetscObject*)&XPipeoldPT);CHKERRQ(ierr);
  if (!XPipeoldPT) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Failed to find composed Vec required by FormResidualPipePT()");
  
  /* get the residual array */
  ierr = DMDAVecGetArray(user->daPipe,RPipePT,&rpipe);CHKERRQ(ierr);
  
  /* get the current guess of solution */
  ierr = DMDAVecGetArrayRead(user->daPipe,XPipePT,&xpipe);CHKERRQ(ierr);
  
  ierr = DMDAVecGetArray(user->daP,user->XP,&xp);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(user->daPaux,user->Paux,&paux);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(user->daHaux,user->Haux,&haux);CHKERRQ(ierr);
  
  ierr = DMDAVecGetArray(user->daPipe,XPipeoldPT,&xpipeoldPT);CHKERRQ(ierr);
  
  /* get dimensions of LOCAL piece of grid (in parallel, this is not full grid) */
  ierr = DMDAGetGridInfo(user->daP,&is,0,0,&ie,0,0,0,0,0,0);CHKERRQ(ierr);
  
  /* apply boundary condition at r=0 */
  if (is==0) {
    i = 0; is++;
    rpipe[i].qp = xpipe[i].qp;
    rpipe[i].Tp = xpipe[i].Tp - haux[i].T;
    rpipe[i].cp = xpipe[i].cp - haux[i].cl;
  }
  
  /* apply boundary condition at r=r_max */
  if (ie==p->ni) {
     i = p->ni-1; ie--;
     rpipe[i].qp = xpipe[i].qp - xpipe[i-1].qp;
     rpipe[i].Tp = xpipe[i].Tp - xpipe[i-1].Tp;
     rpipe[i].cp = xpipe[i].cp - xpipe[i-1].cp;
  }
  
  /* interior of the domain */
  for (i=is; i<p->ni-1; i++) {
    rpipe[i].qp = qpResidual(user,xpipe,xp,haux,paux,i);
    if (p->bulk_comp > 0 && p->bulk_comp < 1) {
      rpipe[i].Tp = TpResidual(user,xpipe,xp,haux,paux,i);
      rpipe[i].cp = cpResidual(user,xpipe,xp,haux,paux,i);
    } else {
      rpipe[i].Tp = xpipe[i].Tp - p->T_A*(p->bulk_comp==1) - p->T_B*(p->bulk_comp==0);
      rpipe[i].cp = xpipe[i].cp - p->bulk_comp;
    }

    rpipe[i].qp += (xpipe[i].qp - xpipeoldPT[i].qp)/p->dtau;
    rpipe[i].Tp += (xpipe[i].Tp - xpipeoldPT[i].Tp)/p->dtau;
    rpipe[i].cp += (xpipe[i].cp - xpipeoldPT[i].cp)/p->dtau;
  }
  
  /* clean up */
  ierr = DMDAVecRestoreArrayRead(user->daPipe,XPipePT,&xpipe);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(user->daP,user->XP,&xp);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(user->daHaux,user->Haux,&haux);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(user->daPaux,user->Paux,&paux);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(user->daPipe,RPipePT,&rpipe);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(user->daPipe,XPipeoldPT,&xpipeoldPT);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}


/* ------------------------------------------------------------------- */
/* Create the compaction pressure residual */
PetscReal PressureResidual(AppCtx *user, const FieldP *xp, FieldPipe *xpipe, AuxFieldH *haux, AuxFieldP *paux, PetscInt i)
/* ------------------------------------------------------------------- */
{
  Parameter   *p = user->param;
  PetscReal   dr = (1-p->base)/(p->ni-2);
  PetscReal   ft, t, E;
  PetscReal   perm_p, perm_m, div_q, residual;
  
  /* if no porosity, compaction pressure undefined so set to zero */
  if (haux[i].phi == 0) { return xp[i].P; }
  
  /* Use spline interpolation to go from E = 0  to E = linear over region delta above Pc */
  t = (xp[i].P - p->Pc)/p->delta_t;
  if (t<=0) {
    ft = 0;
  } else if (0<t && t<1) {
    ft = 1.0 - pow(1.0-t,2);
  } else {
    ft = 1.0;
  }
  E = p->nu*(xp[i].P - p->Pc) *ft;
  
  /* Permeability at the above and below cell faces, geometric mean */
  perm_p = pow(haux[i].phi*haux[i+1].phi,p->perm/2.0);
  perm_m = pow(haux[i].phi*haux[i-1].phi,p->perm/2.0);
  
  /* div(phi^n *((1-phi) - delta*grad(P) )) */
  div_q = (pow(0.5*(paux[i].r+paux[i+1].r),2)*perm_p*(1.0 - p->phi_0*0.5*(haux[i].phi+haux[i+1].phi) - p->delta*(xp[i+1].P-xp[i].P)/dr)
        -  pow(0.5*(paux[i].r+paux[i-1].r),2)*perm_m*(1.0 - p->phi_0*0.5*(haux[i].phi+haux[i-1].phi) - p->delta*(xp[i].P-xp[i-1].P)/dr))
        / (paux[i].r*paux[i].r*dr);
  
  /* construct residual */
  residual = haux[i].phi*xp[i].P + div_q + E;
  
  return(residual);
}

/* ------------------------------------------------------------------- */
/* Create the enthalpy residual */
PetscReal EnthalpyResidual(AppCtx *user, const FieldH *xh, FieldH *xhold, FieldH *xhtheta, FieldP *xptheta, FieldPipe *xpipetheta, AuxFieldH *hauxtheta, AuxFieldP *pauxtheta, PetscInt i)
/* ------------------------------------------------------------------- */
{
  Parameter   *p = user->param;
  PetscReal   dr = (1.0-p->base)/(p->ni-2);
  PetscReal   v_Fromm_p, v_Fromm_m;
  PetscReal   adv_Ts, adv_Tl, dif_T;
  PetscReal   ft, gphi, gxi, t, phi, xi, hhat_eff, alpha, E, M;
  PetscReal   adv_L;
  PetscReal   residual;
  
  /* Use spline interpolation to go from E = 0  to E = linear over region delta above Pc */
  t = (xptheta[i].P - p->Pc)/p->delta_t;
  if (t<=0) {
    ft = 0;
  } else if (0<t && t<1) {
    ft = 1.0 - pow(1.0-t,2);
  } else {
    ft = 1.0;
  }
  E = p->nu*(xptheta[i].P - p->Pc) *ft;
  
  /* Use spline interpolation to switch off M when qp == 0, and when T < Te */
  phi = xpipetheta[i-1].qp/p->delta_phi;
  if (phi<=0) {
    gphi = 0;
  } else if (0<phi && phi<1) {
    gphi = phi*phi*(3.0-2.0*phi);
  } else {
    gphi = 1.0;
  }
  xi = (hauxtheta[i].T - p->Te)/p->delta_xi;
  if (xi<=0) {
    gxi = 0;
  } else if (0<xi && xi<1) {
    gxi = xi*xi*(3.0-2.0*xi);
  } else {
    gxi = 1.0;
  }
  /* effective emplacement = ha in crust and hb in mantle */
  alpha = (p->T_A-hauxtheta[i].T)/p->delta_alpha;
  if (alpha<=0) {
    hhat_eff = p->hhat_B;
  } else if (0<alpha && alpha<1) {
    hhat_eff = alpha*alpha*(3.0-2.0*alpha)*(p->hhat_A-p->hhat_B) + p->hhat_B;
  } else {
    hhat_eff = p->hhat_A;
  }
  M = hhat_eff*(xpipetheta[i].Tp - hauxtheta[i].T) *gphi*gxi;
  
  /* advection of sensible heat by the solid: div(uT), upwind */
  adv_Ts = (pow(0.5*(pauxtheta[i].r+pauxtheta[i+1].r),2)*pauxtheta[i].u  *hauxtheta[i+1].T
         -  pow(0.5*(pauxtheta[i].r+pauxtheta[i-1].r),2)*pauxtheta[i-1].u*hauxtheta[i].T)
        /(pauxtheta[i].r*pauxtheta[i].r*dr);
  
  /* advection of sensible heat by the liquid: div((qT) */
  adv_Tl = (pow(0.5*(pauxtheta[i].r+pauxtheta[i+1].r),2)*pauxtheta[i].q  *hauxtheta[i].T
         -  pow(0.5*(pauxtheta[i].r+pauxtheta[i-1].r),2)*pauxtheta[i-1].q*hauxtheta[i-1].T)
        /(pauxtheta[i].r*pauxtheta[i].r*dr);
  
  /* advection of latent heat: div(phi_0*phi*u + q) */
  adv_L = (pow(0.5*(pauxtheta[i].r+pauxtheta[i+1].r),2)*(pauxtheta[i].u  *p->phi_0*0.5*(hauxtheta[i].phi+hauxtheta[i+1].phi) + pauxtheta[i].q)
        -  pow(0.5*(pauxtheta[i].r+pauxtheta[i-1].r),2)*(pauxtheta[i-1].u*p->phi_0*0.5*(hauxtheta[i].phi+hauxtheta[i-1].phi) + pauxtheta[i-1].q))
        / (pauxtheta[i].r*pauxtheta[i].r*dr);
  
  /* heat diffusion: div(grad T) */
  dif_T = (pow(0.5*(pauxtheta[i].r+pauxtheta[i+1].r),2)*(hauxtheta[i+1].T-hauxtheta[i].T  )
        -  pow(0.5*(pauxtheta[i].r+pauxtheta[i-1].r),2)*(hauxtheta[i].T  -hauxtheta[i-1].T))
        / (pauxtheta[i].r*pauxtheta[i].r*dr*dr);
  
  /* Fromm scheme in middle of domain */
  if (i>1 && i<p->ni-2) {
    /* advection of sensible heat by the solid: div(uT) */
    v_Fromm_p = pow(0.5*(pauxtheta[i].r+pauxtheta[i+1].r),2)*pauxtheta[i].u;
    v_Fromm_m = pow(0.5*(pauxtheta[i].r+pauxtheta[i-1].r),2)*pauxtheta[i-1].u;
    adv_Ts = Fromm_advection(user, v_Fromm_p, v_Fromm_m, hauxtheta[i+2].T, hauxtheta[i+1].T, hauxtheta[i].T, hauxtheta[i-1].T, hauxtheta[i-2].T, pauxtheta[i].r);

    /* advection of sensible heat by the liquid: div(qT) */
    v_Fromm_p = pow(0.5*(pauxtheta[i].r+pauxtheta[i+1].r),2)*pauxtheta[i].q;
    v_Fromm_m = pow(0.5*(pauxtheta[i].r+pauxtheta[i-1].r),2)*pauxtheta[i-1].q;
    adv_Tl = Fromm_advection(user, v_Fromm_p, v_Fromm_m, hauxtheta[i+2].T, hauxtheta[i+1].T, hauxtheta[i].T, hauxtheta[i-1].T, hauxtheta[i-2].T, pauxtheta[i].r);
  }
  
  /* Construct residual */
  residual = (xh[i].H - xhold[i].H)/p->dt + adv_Ts + adv_Tl + p->St*adv_L - dif_T/p->Pe - p->St*p->psi + E*(hauxtheta[i].T + p->St) - M*(xpipetheta[i].Tp + p->St);
  
  return(residual);
}

/* ------------------------------------------------------------------- */
/* Create the composition residual */
PetscReal CompositionResidual(AppCtx *user, const FieldH *xh, FieldH *xhold, FieldH *xhtheta, FieldP *xptheta, FieldPipe *xpipetheta, AuxFieldH *hauxtheta, AuxFieldP *pauxtheta, PetscInt i)
/* ------------------------------------------------------------------- */
{
  Parameter   *p = user->param;
  PetscReal   dr = (1.0-p->base)/(p->ni-2);
  PetscReal   v_Fromm_p, v_Fromm_m;
  PetscReal   adv_liq, adv_sol, dif_l, dif_s;
  PetscReal   ft, gphi, gxi, t, phi, xi, hhat_eff, alpha, E, M;
  PetscReal   residual;
  
  /* Use spline interpolation to go from E = 0  to E = linear over region delta above Pc */
  t = (xptheta[i].P - p->Pc)/p->delta_t;
  if (t<=0) {
    ft = 0;
  } else if (0<t && t<1) {
    ft = 1.0 - pow(1.0-t,2);
  } else {
    ft = 1.0;
  }
  E = p->nu*(xptheta[i].P - p->Pc) *ft;
  
  /* Use spline interpolation to switch off M when qp == 0, and when T < Te */
  phi = xpipetheta[i-1].qp/p->delta_phi;
  if (phi<=0) {
    gphi = 0;
  } else if (0<phi && phi<1) {
    gphi = phi*phi*(3.0-2.0*phi);
  } else {
    gphi = 1.0;
  }
  xi = (hauxtheta[i].T - p->Te)/p->delta_xi;
  if (xi<=0) {
    gxi = 0;
  } else if (0<xi && xi<1) {
    gxi = xi*xi*(3.0-2.0*xi);
  } else {
    gxi = 1.0;
  }
  /* effective emplacement = ha in crust and hb in mantle */
  alpha = (p->T_A-hauxtheta[i].T)/p->delta_alpha;
  if (alpha<=0) {
    hhat_eff = p->hhat_B;
  } else if (0<alpha && alpha<1) {
    hhat_eff = alpha*alpha*(3.0-2.0*alpha)*(p->hhat_A-p->hhat_B) + p->hhat_B;
  } else {
    hhat_eff = p->hhat_A;
  }
  M = hhat_eff*(xpipetheta[i].Tp - hauxtheta[i].T) *gphi*gxi;
  
  /* advection of liquid composition: div((phi_0*phi*u + q)cl) */
  adv_liq = (pow(0.5*(pauxtheta[i].r+pauxtheta[i+1].r),2)*(p->phi_0*0.5*(hauxtheta[i].phi+hauxtheta[i+1].phi)*pauxtheta[i].u   + pauxtheta[i].q) *0.5*(hauxtheta[i].cl+hauxtheta[i+1].cl)
           - pow(0.5*(pauxtheta[i].r+pauxtheta[i-1].r),2)*(p->phi_0*0.5*(hauxtheta[i].phi+hauxtheta[i-1].phi)*pauxtheta[i-1].u + pauxtheta[i-1].q) *0.5*(hauxtheta[i].cl+hauxtheta[i-1].cl))
             /(pauxtheta[i].r*pauxtheta[i].r*dr);
  
  /* advection of solid composition: div((1-phi_0*phi)*u*cs) */
  adv_sol = (pow(0.5*(pauxtheta[i].r+pauxtheta[i+1].r),2)*(1 - p->phi_0*0.5*(hauxtheta[i].phi+hauxtheta[i+1].phi))*pauxtheta[i].u  *0.5*(hauxtheta[i].cs+hauxtheta[i+1].cs)
           - pow(0.5*(pauxtheta[i].r+pauxtheta[i-1].r),2)*(1 - p->phi_0*0.5*(hauxtheta[i].phi+hauxtheta[i-1].phi))*pauxtheta[i-1].u*0.5*(hauxtheta[i].cs+hauxtheta[i-1].cs))
             /(pauxtheta[i].r*pauxtheta[i].r*dr);
  
  /* liquid composition diffusion: div(phi_0*phi grad(cl)) */
  dif_l = (pow(0.5*(pauxtheta[i].r+pauxtheta[i+1].r),2)*p->phi_0*0.5*(hauxtheta[i].phi+hauxtheta[i+1].phi)*(hauxtheta[i+1].cl-hauxtheta[i].cl )
      -  pow(0.5*(pauxtheta[i].r+pauxtheta[i-1].r),2)*p->phi_0*0.5*(hauxtheta[i].phi+hauxtheta[i-1].phi)*(hauxtheta[i].cl  -hauxtheta[i-1].cl))
      / (pauxtheta[i].r*pauxtheta[i].r*dr*dr);
  dif_s = (pow(0.5*(pauxtheta[i].r+pauxtheta[i+1].r),2)*(1.0 - p->phi_0*0.5*(hauxtheta[i].phi+hauxtheta[i+1].phi))*(hauxtheta[i+1].cs-hauxtheta[i].cs )
  -  pow(0.5*(pauxtheta[i].r+pauxtheta[i-1].r),2)*(1.0 - p->phi_0*0.5*(hauxtheta[i].phi+hauxtheta[i-1].phi))*(hauxtheta[i].cs  -hauxtheta[i-1].cs))
  / (pauxtheta[i].r*pauxtheta[i].r*dr*dr);
  
  /* Fromm scheme in middle of domain */
  if (i>1 && i<p->ni-2) {
    /* advection of liquid composition: div((phi_0*phi*u + q)cl) */
    v_Fromm_p = pow(0.5*(pauxtheta[i].r+pauxtheta[i+1].r),2)*(p->phi_0*0.5*(hauxtheta[i].phi+hauxtheta[i+1].phi)*pauxtheta[i].u   + pauxtheta[i].q);
    v_Fromm_m = pow(0.5*(pauxtheta[i].r+pauxtheta[i-1].r),2)*(p->phi_0*0.5*(hauxtheta[i].phi+hauxtheta[i-1].phi)*pauxtheta[i-1].u + pauxtheta[i-1].q);
    adv_liq = Fromm_advection(user, v_Fromm_p, v_Fromm_m, hauxtheta[i+2].cl, hauxtheta[i+1].cl, hauxtheta[i].cl, hauxtheta[i-1].cl, hauxtheta[i-2].cl, pauxtheta[i].r);

    /* advection of solid composition: div((1-phi_0*phi)*u*cs) */
    v_Fromm_p = pow(0.5*(pauxtheta[i].r+pauxtheta[i+1].r),2)*(1 - p->phi_0*0.5*(hauxtheta[i].phi+hauxtheta[i+1].phi))*pauxtheta[i].u;
    v_Fromm_m = pow(0.5*(pauxtheta[i].r+pauxtheta[i-1].r),2)*(1 - p->phi_0*0.5*(hauxtheta[i].phi+hauxtheta[i-1].phi))*pauxtheta[i-1].u;
    adv_sol = Fromm_advection(user, v_Fromm_p, v_Fromm_m, hauxtheta[i+2].cs, hauxtheta[i+1].cs, hauxtheta[i].cs, hauxtheta[i-1].cs, hauxtheta[i-2].cs, pauxtheta[i].r);
  }
  
  /* Construct residual */
  residual = (xh[i].c - xhold[i].c)/p->dt + adv_liq + adv_sol - dif_l/p->Pec - dif_s/p->Pec + E*hauxtheta[i].cl - M*xpipetheta[i].cp;
  
  return(residual);
}

/* ------------------------------------------------------------------- */
/* Create the plumbing system flux residual */
PetscReal qpResidual(AppCtx *user, const FieldPipe *xpipe, FieldP *xp, AuxFieldH *haux, AuxFieldP *paux, PetscInt i)
/* ------------------------------------------------------------------- */
{
  Parameter   *p = user->param;
  PetscReal   dr = (1.0-p->base)/(p->ni-2);
  PetscReal   div_qp;
  PetscReal   ft, gphi, gxi, t, phi, xi, hhat_eff, alpha, E, M;
  PetscReal   residual;
  
  /* Use spline interpolation to go from E = 0  to E = linear over region delta above Pc */
  t = (xp[i].P - p->Pc)/p->delta_t;
  if (t<=0) {
    ft = 0;
  } else if (0<t && t<1) {
    ft = 1.0 - pow(1.0-t,2);
  } else {
    ft = 1.0;
  }
  E = p->nu*(xp[i].P - p->Pc) *ft;
  
  /* Use spline interpolation to switch off M when E !=0, when qp == 0, and when T < Te */
  phi = xpipe[i-1].qp/p->delta_phi;
  if (phi<=0) {
    gphi = 0;
  } else if (0<phi && phi<1) {
    gphi = phi*phi*(3.0-2.0*phi);
  } else {
    gphi = 1.0;
  }
  xi = (haux[i].T - p->Te)/p->delta_xi;
  if (xi<=0) {
    gxi = 0;
  } else if (0<xi && xi<1) {
    gxi = xi*xi*(3.0-2.0*xi);
  } else {
    gxi = 1.0;
  }
  /* effective emplacement = ha in crust and hb in mantle */
  alpha = (p->T_A-haux[i].T)/p->delta_alpha;
  if (alpha<=0) {
    hhat_eff = p->hhat_B;
  } else if (0<alpha && alpha<1) {
    hhat_eff = alpha*alpha*(3.0-2.0*alpha)*(p->hhat_A-p->hhat_B) + p->hhat_B;
  } else {
    hhat_eff = p->hhat_A;
  }
  M = hhat_eff*(xpipe[i].Tp - haux[i].T) *gphi*gxi;
  
  div_qp = (pow(0.5*(paux[i].r+paux[i+1].r),2)*xpipe[i].qp
          - pow(0.5*(paux[i].r+paux[i-1].r),2)*xpipe[i-1].qp)/(paux[i].r*paux[i].r*dr);
  
  residual = div_qp - E + M;
  
  return(residual);
}

/* ------------------------------------------------------------------- */
/* Create the plumbing system temperature residual */
PetscReal TpResidual(AppCtx *user, const FieldPipe *xpipe, FieldP *xp, AuxFieldH *haux, AuxFieldP *paux, PetscInt i)
/* ------------------------------------------------------------------- */
{
  Parameter   *p = user->param;
  PetscReal   dr = (1.0-p->base)/(p->ni-2);
  PetscReal   v_Fromm_p, v_Fromm_m;
  PetscReal   adv_Tp;
  PetscReal   ft, gphi, gxi, t, phi, xi, hhat_eff, alpha, E, M, eps = 1e-7;
  PetscReal   residual;
  
  /* Use spline interpolation to go from E = 0  to E = linear over region delta above Pc */
  t = (xp[i].P - p->Pc)/p->delta_t;
  if (t<=0) {
    ft = 0;
  } else if (0<t && t<1) {
    ft = 1.0 - pow(1.0-t,2);
  } else {
    ft = 1.0;
  }
  E = p->nu*(xp[i].P - p->Pc) *ft;
  
  /* Use spline interpolation to switch off M when qp == 0, and when T < Te */
  phi = xpipe[i-1].qp/p->delta_phi;
  if (phi<=0) {
    gphi = 0;
  } else if (0<phi && phi<1) {
    gphi = phi*phi*(3.0-2.0*phi);
  } else {
    gphi = 1.0;
  }
  xi = (haux[i].T - p->Te)/p->delta_xi;
  if (xi<=0) {
    gxi = 0;
  } else if (0<xi && xi<1) {
    gxi = xi*xi*(3.0-2.0*xi);
  } else {
    gxi = 1.0;
  }
  /* effective emplacement = ha in crust and hb in mantle */
  alpha = (p->T_A-haux[i].T)/p->delta_alpha;
  if (alpha<=0) {
    hhat_eff = p->hhat_B;
  } else if (0<alpha && alpha<1) {
    hhat_eff = alpha*alpha*(3.0-2.0*alpha)*(p->hhat_A-p->hhat_B) + p->hhat_B;
  } else {
    hhat_eff = p->hhat_A;
  }
  M = hhat_eff*(xpipe[i].Tp - haux[i].T) *gphi*gxi;
  
  adv_Tp = (pow(0.5*(paux[i].r+paux[i+1].r),2)*xpipe[i].qp  *xpipe[i].Tp
          - pow(0.5*(paux[i].r+paux[i-1].r),2)*xpipe[i-1].qp*xpipe[i-1].Tp)/(paux[i].r*paux[i].r*dr);
  
  /* Upwind scheme in middle of domain */
  if (i>1 && i<p->ni-2) {
    /* advection of liquid composition: div((phi_0*phi*u + q)cl) */
    v_Fromm_p = pow(0.5*(paux[i].r+paux[i+1].r),2)*xpipe[i].qp;
    v_Fromm_m = pow(0.5*(paux[i].r+paux[i-1].r),2)*xpipe[i-1].qp;
    adv_Tp = Fromm_advection(user, v_Fromm_p, v_Fromm_m, xpipe[i+2].Tp, xpipe[i+1].Tp, xpipe[i].Tp, xpipe[i-1].Tp, xpipe[i-2].Tp, paux[i].r);
  }
  
  residual = adv_Tp - E*haux[i].T + M*xpipe[i].Tp + eps*(xpipe[i].Tp-haux[i].T); // final term is a reularisation so that there is always a small amount of xpipe derivative, even when E, M, and qp = 0.
  
  return(residual);
}

/* ------------------------------------------------------------------- */
/* Create the plumbing system composition residual */
PetscReal cpResidual(AppCtx *user, const FieldPipe *xpipe, FieldP *xp, AuxFieldH *haux, AuxFieldP *paux, PetscInt i)
/* ------------------------------------------------------------------- */
{
  Parameter   *p = user->param;
  PetscReal   dr = (1.0-p->base)/(p->ni-2);
  PetscReal   v_Fromm_p, v_Fromm_m;
  PetscReal   adv_cp;
  PetscReal   ft, gphi, gxi, t, phi, xi, E, M, hhat_eff, alpha, eps = 1e-7;
  PetscReal   residual;
  
  /* Use spline interpolation to go from E = 0  to E = linear over region delta above Pc */
  t = (xp[i].P - p->Pc)/p->delta_t;
  if (t<=0) {
    ft = 0;
  } else if (0<t && t<1) {
    ft = 1.0 - pow(1.0-t,2);
  } else {
    ft = 1.0;
  }
  E = p->nu*(xp[i].P - p->Pc) *ft;
  
  /* Use spline interpolation to switch off M when qp == 0, and when T < Te */
  phi = xpipe[i-1].qp/p->delta_phi;
  if (phi<=0) {
    gphi = 0;
  } else if (0<phi && phi<1) {
    gphi = phi*phi*(3.0-2.0*phi);
  } else {
    gphi = 1.0;
  }
  xi = (haux[i].T - p->Te)/p->delta_xi;
  if (xi<=0) {
    gxi = 0;
  } else if (0<xi && xi<1) {
    gxi = xi*xi*(3.0-2.0*xi);
  } else {
    gxi = 1.0;
  }
  /* effective emplacement = ha in crust and hb in mantle */
  alpha = (p->T_A-haux[i].T)/p->delta_alpha;
  if (alpha<=0) {
    hhat_eff = p->hhat_B;
  } else if (0<alpha && alpha<1) {
    hhat_eff = alpha*alpha*(3.0-2.0*alpha)*(p->hhat_A-p->hhat_B) + p->hhat_B;
  } else {
    hhat_eff = p->hhat_A;
  }
  M = hhat_eff*(xpipe[i].Tp - haux[i].T) *gphi*gxi;
  
  adv_cp = (pow(0.5*(paux[i].r+paux[i+1].r),2)*xpipe[i].qp  *xpipe[i].cp
          - pow(0.5*(paux[i].r+paux[i-1].r),2)*xpipe[i-1].qp*xpipe[i-1].cp)/(paux[i].r*paux[i].r*dr);
  
  /* Upwind scheme in middle of domain */
  if (i>1 && i<p->ni-2) {
    /* advection of liquid composition: div((phi_0*phi*u + q)cl) */
    v_Fromm_p = pow(0.5*(paux[i].r+paux[i+1].r),2)*xpipe[i].qp;
    v_Fromm_m = pow(0.5*(paux[i].r+paux[i-1].r),2)*xpipe[i-1].qp;
    adv_cp = Fromm_advection(user, v_Fromm_p, v_Fromm_m, xpipe[i+2].cp, xpipe[i+1].cp, xpipe[i].cp, xpipe[i-1].cp, xpipe[i-2].cp, paux[i].r);
  }
  
  residual = adv_cp - E*haux[i].cl + M*xpipe[i].cp + eps*(xpipe[i].cp-haux[i].cl); // final term is a reularisation so that there is always a small amount of xpipe derivative, even when E, M, and qp = 0.
  
  return(residual);
}

/* ------------------------------------------------------------------- */
/* Calculates the auxilliary parameters associated with the pressure solver */
PetscErrorCode PAuxParamsCalc(AppCtx *user, const FieldP *xp, FieldPipe *xpipe, AuxFieldH *haux, AuxFieldP *paux)
/* ------------------------------------------------------------------- */
{
  Parameter   *p = user->param;
  PetscReal   dr = (1.0-p->base)/(p->ni-2);
  PetscInt    i;
  PetscReal   u_max=0;
  PetscFunctionBegin;
  
  for(i=0; i<p->ni; i++) {
    /* set Darcy flux from Darcy's Law, q = (phi_0*phi)^n *(1-phi_0*phi - delta grad(P)) */
    /* avoid overrunning end of arrays by setting surface q = 0 */
    if (i!=p->ni-1) {
      paux[i].q = pow(haux[i].phi*haux[i+1].phi,p->perm/2)*(1.0 - p->phi_0*0.5*(haux[i].phi+haux[i+1].phi) - p->delta*(xp[i+1].P-xp[i].P)/dr);
    } else {
      paux[i].q = 0;
    }
    
    /* solid velocity from continuity, only true in 1D */
    paux[i].u = -(xpipe[i].qp + paux[i].q);
    
    if(fabs(paux[i].u)>u_max) {
      u_max = fabs(paux[i].u);
      p->CFL = u_max*(p->psi_0*p->R/(p->rho_0*p->L))*p->dt*(p->rho_0*p->L/p->psi_0)/(dr*p->R);
    }

  }
  
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
/* Calculates the auxilliary parameters associated with the enthalpy solver */
PetscErrorCode HAuxParamsCalc(AppCtx *user, const FieldH *xh, AuxFieldH *haux)
/* ------------------------------------------------------------------- */
{
  Parameter   *p = user->param;
  PetscInt    i;
  PetscInt    newt_its = 0, max_newt_its = 50;
  PetscBool   newt_converged = PETSC_FALSE;
  PetscReal   phi_newt_tol = 1e-10, phi_resid =1;
  PetscReal   dcl_dphi, dcs_dphi, dresid;
  PetscFunctionBegin;

  for(i=0; i<p->ni; i++) {

    /* If at end-member composition, or below local solidus, can solve for phi, T, cl, cs. Else do newton iteration */
    if (
        (xh[i].c > 0) &&
        (xh[i].c < 1) &&
        (xh[i].H > p->T_B - (p->T_B-p->T_A)*(1.0-exp(-xh[i].c/p->gamma))/(1.0-exp(-1.0/p->gamma)))
        ) {

      /* take first guess of phi will be that of previous timestep or iteration*/
      newt_its = 0;
      
      haux[i].T = xh[i].H - p->St*p->phi_0*haux[i].phi; // T guess from H = T + St*phi_0*phi
      haux[i].cl = (p->T_B - haux[i].T)/(p->T_B - p->T_A); // cl guess from cl = (TB-T)/(TB-TA)
      haux[i].cs = (xh[i].c - p->phi_0*haux[i].phi*haux[i].cl)/(1.0-p->phi_0*haux[i].phi); // create solid composition guess from c = phi_0*phi*cl + (1-phi_0*phi)*cs
      phi_resid = p->T_B - haux[i].T - (p->T_B-p->T_A)*(1.0-exp(-haux[i].cs/p->gamma))/(1.0-exp(-1.0/p->gamma)); // create residual associated with this from solidus curve equation T = TB - (TB-TA)(1-exp(-cs/gamma))/(1-exp(-1/gamma))
      //printf("  ** #local newton %d : |F| %+1.4e\n",newt_its,phi_resid);
      
      for (newt_its=1; newt_its<=max_newt_its; newt_its++) {
        
        /* calculate derivative of residual wrt phi */
        dcl_dphi = p->St*p->phi_0/(p->T_B - p->T_A); // gradient correct so Tl = Ts at c=1.
        dcs_dphi = xh[i].c*p->phi_0/pow(1.0-p->phi_0*haux[i].phi,2) - p->phi_0*haux[i].cl/(1.0-p->phi_0*haux[i].phi) - p->phi_0*haux[i].phi*dcl_dphi/(1.0-p->phi_0*haux[i].phi) - pow(p->phi_0,2)*haux[i].phi*haux[i].cl/(pow(1.0-p->phi_0*haux[i].phi,2));
        dresid = p->St*p->phi_0 - (p->T_B-p->T_A)*dcs_dphi*exp(-haux[i].cs/p->gamma)/(p->gamma*(1.0-exp(-1.0/p->gamma)));
        
        haux[i].phi = haux[i].phi - phi_resid/dresid; // update phi using Newton method
        
        haux[i].T = xh[i].H - p->St*p->phi_0*haux[i].phi; // T guess from H = T + St*phi_0*phi
        haux[i].cl = (p->T_B - haux[i].T)/(p->T_B - p->T_A); // cl guess from cl = (TB-T)/(TB-TA)
        haux[i].cs = (xh[i].c - p->phi_0*haux[i].phi*haux[i].cl)/(1.0-p->phi_0*haux[i].phi); // create solid composition guess from c = phi_0*phi*cl + (1-phi_0*phi)*cs
        phi_resid = p->T_B - haux[i].T - (p->T_B-p->T_A)*(1.0-exp(-haux[i].cs/p->gamma))/(1.0-exp(-1.0/p->gamma)); // create residual associated with this from solidus curve equation T = TB - (TB-TA)(1-exp(-cs/gamma))/(1-exp(-1/gamma))
        
        if(PetscAbsReal(phi_resid) < phi_newt_tol) {
          newt_converged = PETSC_TRUE;
          break;
        }
        
      }
      if (!newt_converged) {
        PetscPrintf(user->comm,"Max its reached and convergence wasn't achievd is greater than eps\n");
        SETERRQ(user->comm,PETSC_ERR_SUP,"Cannot proceed without a converged porosity solution");
      }
      
    } else {
      /* if below local solidus */
      if (xh[i].H <= p->T_B - (p->T_B-p->T_A)*(1.0-exp(-xh[i].c/p->gamma))/(1.0-exp(-1.0/p->gamma))) {
        haux[i].T = xh[i].H;
        haux[i].phi = 0;
        haux[i].cl = xh[i].c;
        haux[i].cs = xh[i].c;
      } else {
        /* above solidus, below liquidus but at end member composition */
        if (xh[i].c >= 1.0) {
          haux[i].T = p->T_A;
          haux[i].phi = (xh[i].H-p->T_A)/(p->St*p->phi_0);
          haux[i].cl = 1.0;
          haux[i].cs = 1.0;
        }
        if (xh[i].c <= 0) {
          haux[i].T = p->T_B;
          haux[i].phi = (xh[i].H-p->T_B)/(p->St*p->phi_0);
          haux[i].cl = 0;
          haux[i].cs = 0;
        }
      }
    }
  }
  
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
/* Check periodically whether steady state has been achieved */
PetscBool SteadyStateChecker(AppCtx *user)
/* ------------------------------------------------------------------- */
{
  Parameter   *p = user->param;
  PetscBool   STEADY = PETSC_TRUE;
  Vec         XP_sum, XH_sum, XPipe_sum;
  PetscReal   XP_L2, XH_L2, XPipe_L2;
  PetscScalar a = -1, b = 1;
  PetscErrorCode ierr;
  
  ierr = VecDuplicate(user->XP,&XP_sum); CHKERRQ(ierr);
  ierr = VecCopy(user->XP,XP_sum); CHKERRQ(ierr);
  ierr = VecDuplicate(user->XH,&XH_sum); CHKERRQ(ierr);
  ierr = VecCopy(user->XH,XH_sum); CHKERRQ(ierr);
  ierr = VecDuplicate(user->XPipe,&XPipe_sum); CHKERRQ(ierr);
  ierr = VecCopy(user->XPipe,XPipe_sum); CHKERRQ(ierr);
  
  ierr = VecAXPBY(XP_sum,a,b,user->XPold);CHKERRQ(ierr);
  ierr = VecAXPBY(XH_sum,a,b,user->XHold);CHKERRQ(ierr);
  ierr = VecAXPBY(XPipe_sum,a,b,user->XPipeold);CHKERRQ(ierr);
  
  ierr = VecNorm(XP_sum,NORM_2,&XP_L2);CHKERRQ(ierr);
  ierr = VecNorm(XH_sum,NORM_2,&XH_L2);CHKERRQ(ierr);
  ierr = VecNorm(XP_sum,NORM_2,&XPipe_L2);CHKERRQ(ierr);
  
  PetscPrintf(user->comm,"SS checker - XP_L2 = %e, XH_L2 = %e, XPipe_L2 = %e \n",XP_L2,XH_L2,XPipe_L2);
  if (XP_L2 < p->steady_tol && XH_L2 < p->steady_tol && XPipe_L2 < p->steady_tol) {
    STEADY = PETSC_TRUE;
  } else {
    STEADY = PETSC_FALSE;
  }
  
  return(STEADY);
}

/* ------------------------------------------------------------------- */
/* Calculates a flux divergence term using the upwind Fromm scheme (spherical) */
PetscReal Fromm_advection(AppCtx *user, PetscReal v_p, PetscReal v_m,
        PetscReal q_p2, PetscReal q_p, PetscReal q_c,
        PetscReal q_m, PetscReal q_m2, PetscReal r)
/* ------------------------------------------------------------------- */
{
  Parameter   *p = user->param;
  PetscReal   dr = (1.0-p->base)/(p->ni-2);
  PetscReal   adv_term;
  
  adv_term = ( (v_p/8 *(-q_p2 + 5*(q_p + q_c) - q_m ) - fabs(v_p)/8 *(-q_p2 + 3*(q_p - q_c) + q_m) )
     -   (v_m/8 *(-q_p  + 5*(q_c + q_m) - q_m2) - fabs(v_m)/8 *(-q_p  + 3*(q_c - q_m) + q_m2)))/(dr*r*r);
  
  return(adv_term);
}

/* ------------------------------------------------------------------- */
/* Calculate the melting rate, E, and M for outputting and plotting    */
PetscErrorCode Extra_output_params(AppCtx *user)
/* ------------------------------------------------------------------- */
{
  Parameter *p = user->param;
  FieldH  *xh;
  FieldP  *xp;
  FieldPipe *xpipe;
  AuxFieldH *haux, *hauxold;
  AuxFieldP *paux;
  FieldOUT  *xout;
  PetscInt  i;
  PetscReal dr = (1.0-p->base)/(p->ni-2), dif_l;
  PetscReal ft, gphi, gxi, t, xi, phi, hhat_eff, alpha;
  PetscErrorCode ierr;
  
  ierr = DMDAVecGetArray(user->daH,user->XH,&xh);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(user->daP,user->XP,&xp);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(user->daPipe,user->XPipe,&xpipe);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(user->daHaux,user->Haux,&haux);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(user->daHaux,user->Hauxold,&hauxold);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(user->daPaux,user->Paux,&paux);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(user->daOUT,user->XOUT,&xout);CHKERRQ(ierr);
  
  for (i=0;i<p->ni;i++) {
    /* Use spline interpolation to go from E = 0  to E = linear over region delta above Pc */
    t = (xp[i].P - p->Pc)/p->delta_t;
    if (t<=0) {
      ft = 0;
    } else if (0<t && t<1) {
      ft = 1.0 - pow(1.0-t,2);
    } else {
      ft = 1.0;
    }
    xout[i].E = p->nu*(xp[i].P - p->Pc) *ft;
    
    /* Use spline interpolation to switch off M when qp == 0, and when T < Te */
    phi = xpipe[i-1].qp/p->delta_phi;
    if (phi<=0) {
      gphi = 0;
    } else if (0<phi && phi<1) {
      gphi = phi*phi*(3.0-2.0*phi);
    } else {
      gphi = 1.0;
    }
    xi = (haux[i].T - p->Te)/p->delta_xi;
    if (xi<=0) {
      gxi = 0;
    } else if (0<xi && xi<1) {
      gxi = xi*xi*(3.0-2.0*xi);
    } else {
      gxi = 1.0;
    }
    /* effective emplacement = ha in crust and hb in mantle */
    alpha = (p->T_A-haux[i].T)/p->delta_alpha;
    if (alpha<=0) {
      hhat_eff = p->hhat_B;
    } else if (0<alpha && alpha<1) {
      hhat_eff = alpha*alpha*(3.0-2.0*alpha)*(p->hhat_A-p->hhat_B) + p->hhat_B;
    } else {
      hhat_eff = p->hhat_A;
    }
    xout[i].M = hhat_eff*(xpipe[i].Tp - haux[i].T) *gphi*gxi;

    /* calculate the melting rates */
    if (i>0 && i<p->ni-1 && haux[i].phi > 1e-8 && haux[i+1].phi > 1e-8) {
      xout[i].Gamma = p->phi_0*(haux[i].phi-hauxold[i].phi)/p->dt +
                      (pow(0.5*(paux[i].r+paux[i+1].r),2.0)*(p->phi_0*0.5*(haux[i].phi+haux[i+1].phi)*paux[i].u + paux[i].q) -
                       pow(0.5*(paux[i].r+paux[i-1].r),2.0)*(p->phi_0*0.5*(haux[i].phi+haux[i-1].phi)*paux[i-1].u + paux[i-1].q))/(pow(paux[i].r,2.0)*dr) + xout[i].E;

      dif_l = (pow(0.5*(paux[i].r+paux[i+1].r),2)*p->phi_0*0.5*(haux[i].phi+haux[i+1].phi)*(haux[i+1].cl-haux[i].cl )
          -  pow(0.5*(paux[i].r+paux[i-1].r),2)*p->phi_0*0.5*(haux[i].phi+haux[i-1].phi)*(haux[i].cl  -haux[i-1].cl))
          / (paux[i].r*paux[i].r*dr*dr);

      xout[i].Gamma_A = p->phi_0* (haux[i].phi*haux[i].cl - hauxold[i].phi*hauxold[i].cl)/p->dt +
                        (pow(0.5*(paux[i].r+paux[i+1].r),2.0)*(p->phi_0*0.5*(haux[i].phi+haux[i+1].phi)*paux[i].u + paux[i].q)*0.5*(haux[i].cl+haux[i+1].cl) -
                         pow(0.5*(paux[i].r+paux[i-1].r),2.0)*(p->phi_0*0.5*(haux[i].phi+haux[i-1].phi)*paux[i-1].u + paux[i-1].q)*0.5*(haux[i].cl+haux[i-1].cl))/(pow(paux[i].r,2.0)*dr) +
                         xout[i].E*haux[i].cl - dif_l/p->Pec;

    } else {
      xout[i].Gamma = 0; // melting rate will be zero at end nodes (as outside the domain)
      xout[i].Gamma_A = 0;
    }
    
  }
  
  ierr = DMDAVecRestoreArray(user->daH,user->XH,&xh);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(user->daP,user->XP,&xp);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(user->daPipe,user->XPipe,&xpipe);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(user->daHaux,user->Haux,&haux);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(user->daHaux,user->Hauxold,&hauxold);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(user->daPaux,user->Paux,&paux);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(user->daOUT,user->XOUT,&xout);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
/* Set up parameter bag */
PetscErrorCode ParameterSetup(AppCtx user)
/* ------------------------------------------------------------------- */
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  
  /* GRID PARAMETERS */
  ierr = PetscBagRegisterInt(user.bag,&(user.param->ni),4000,"ni","Number of grid points"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(user.bag,&(user.param->ns),100000000,"ns","Number of time steps"); CHKERRQ(ierr);
  
  /* IO STRUCTURE PARAMETERS */
  ierr = PetscBagRegisterReal(user.bag,&(user.param->R_cmb),700000,"R_cmb","Core-mantle boundary (base of domain)"); CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(user.bag,&(user.param->R),1820000,"R","Radius of Io"); CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(user.bag,&(user.param->base),0,"base","Non-dim base of domain, set in initialsation"); CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(user.bag,&(user.param->g),1.5,"g","constant gravity"); CHKERRQ(ierr);
  
  /* IO MATERIAL PARAMETERS */
  ierr = PetscBagRegisterReal(user.bag,&(user.param->perm),3,"perm","Porosity exponent"); CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(user.bag,&(user.param->L),400000,"L","Latent heat of fusion (J/kg)"); CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(user.bag,&(user.param->ce),1200,"ce","specific heat capacity (J/kg/K)"); CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(user.bag,&(user.param->K_0),1e-7,"K_0","Mobility"); CHKERRQ(ierr); // from Katz 2010
  ierr = PetscBagRegisterReal(user.bag,&(user.param->eta),1e20,"eta","reference shear viscosity"); CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(user.bag,&(user.param->eta_l),1,"eta_l","reference liquid viscosity"); CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(user.bag,&(user.param->rho_0),3000,"rho_0","density of mantle"); CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(user.bag,&(user.param->del_rho),500,"del_rho","Boussinesq density difference"); CHKERRQ(ierr); // from Katz 2010
  ierr = PetscBagRegisterReal(user.bag,&(user.param->kappa),1e-6,"kappa","thermal diffusivity"); CHKERRQ(ierr); //value from Katz 2010
  ierr = PetscBagRegisterReal(user.bag,&(user.param->zeta_0),0,"zeta_0","Reference compaction viscosity, set in initialsation"); CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(user.bag,&(user.param->D),1e-8,"D","Chemical diffusivity (m2/s)"); CHKERRQ(ierr); //value from Katz 2010
  
  /* IO NON-DIMENSIONAL PARAMETERS */
  ierr = PetscBagRegisterReal(user.bag,&(user.param->Pe),0,"Pe","Peclet number, set in initialsation"); CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(user.bag,&(user.param->Pec),0,"Pec","Compositional Peclet number, set in initialsation"); CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(user.bag,&(user.param->St),0,"St","Stefan number, set in initialsation"); CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(user.bag,&(user.param->delta),0,"delta","Compaction parameter, set in initialsation"); CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(user.bag,&(user.param->phi_0),0,"phi_0","Reference porosity, necessarily set by reference velocity and hence by tidal heating, set in initialsation"); CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(user.bag,&(user.param->nu),1000,"nu","The constant that sets how rapidly melt is extracted"); CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(user.bag,&(user.param->delta_t),0.01,"delta_t","Associated with the pressure switch on extraction"); CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(user.bag,&(user.param->delta_phi),0.01,"delta_phi","Associated with the flux switch on emplacement"); CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(user.bag,&(user.param->delta_xi),0.01,"delta_xi","Associated with the Te switch on emplacement"); CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(user.bag,&(user.param->delta_alpha),0.001,"delta_alpha","Associated with the switch between ha and hb"); CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(user.bag,&(user.param->hhat_A),100,"hhat_A","Emplacement rate constant for end-member composition A"); CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(user.bag,&(user.param->hhat_B),100,"hhat_B","Emplacement rate constant for end-member composition B"); CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(user.bag,&(user.param->Pc),0.1,"Pc","Critical overpressure above which extraction occurs"); CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(user.bag,&(user.param->P_0),0,"P_0","Reference compaction pressure, calculated in initialisation"); CHKERRQ(ierr);
  
  /* TEMPERATURE AND HEATING PARAMETERS */
  ierr = PetscBagRegisterReal(user.bag,&(user.param->T_0),1500-150,"T_0","reference temperature (high MP component)"); CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(user.bag,&(user.param->T_B),1,"T_B","non-dimensional high MP component melting temp"); CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(user.bag,&(user.param->T_A),0.8,"T_A","non-dimensional low MP component melting temp"); CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(user.bag,&(user.param->T_surf_dim),150,"T_surf_dim","average surface temperature"); CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(user.bag,&(user.param->Te),0,"Te","The emplacement cut-off temperature"); CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(user.bag,&(user.param->Psi_0),1e14,"Psi_0","Input global tidal heating"); CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(user.bag,&(user.param->psi_0),0,"psi_0","Reference constant volumetric heating, set in initialsation"); CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(user.bag,&(user.param->psi),1,"psi","non-dimensional tidal heating rate"); CHKERRQ(ierr);
  
  /* BOUNDARY CONDITION PARAMETERS */
  ierr = PetscBagRegisterReal(user.bag,&(user.param->H_end),0,"H_end","BC for enthalpy at top of domain"); CHKERRQ(ierr);
  
  /* COMPOSITION PARAMETERS */
  ierr = PetscBagRegisterReal(user.bag,&(user.param->gamma),1e-2,"gamma","Controls the shape of the solidus curve");CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(user.bag,&(user.param->bulk_comp),0.5,"bulk_comp","The desired bulk composition, provides the initial condition");CHKERRQ(ierr);

  /* TIMESTEPPING PARAMETERS */
  ierr = PetscBagRegisterReal(user.bag,&(user.param->t),0,"t","(NO SET) Current time");CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(user.bag,&(user.param->dt),1e-3,"dt","Time-step size");CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(user.bag,&(user.param->CFL),0,"CFL","reporting CFL value");CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(user.bag,&(user.param->it),1,"it","holds what timestep we're at"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(user.bag,&(user.param->SNESit),15,"SNESit","controls how many iterations are done within a timestep"); CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(user.bag,&(user.param->tmax),1e10,"tmax","Maximum model time");CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(user.bag,&(user.param->theta),1,"theta","For theta method, 0.5 = CN, 1 = Implcit, 0 = explicit");CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(user.bag,&(user.param->steady_tol),1e-6,"steady_tol","Tolerance for deciding if steady state has been achieved");CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(user.bag,&(user.param->dtau),1,"dtau","Pseudo-timestepp time-step size");CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(user.bag,&(user.param->true_solve_flag),0,"true_solve_flag","flag used in debugging that tells when you're in the true solve"); CHKERRQ(ierr);
  
  /* OUTPUT FILE PARAMETERS */
  ierr = PetscBagRegisterInt(user.bag,&(user.param->out_freq),100,"out_freq","how often to output solution, default 0 = only at end"); CHKERRQ(ierr);
  ierr = PetscBagRegisterString(user.bag,&(user.param->filename),FNAME_LENGTH,"IoComp","filename","Name of output file"); CHKERRQ(ierr);
  ierr = PetscBagRegisterString(user.bag,&(user.param->output_dir),PETSC_MAX_PATH_LEN-1,"outputs/","output_dir","Name of output directory"); CHKERRQ(ierr);
  ierr = PetscBagRegisterString(user.bag,&(user.param->SS_output_dir),PETSC_MAX_PATH_LEN-1,"SS_outputs/","SS_output_dir","Name of output directory"); CHKERRQ(ierr);
  
  /* RESTART PARAMETERS */
  ierr = PetscBagRegisterInt(user.bag,&(user.param->restart_step),0,"restart_step","Step to restart from, 0 = start new, -1 = from steady state file"); CHKERRQ(ierr);
  ierr = PetscBagRegisterString(user.bag,&(user.param->restart_ss),PETSC_MAX_PATH_LEN-1,"SSc050_ha200_hb010_Pc01_Te04","restart_ss","Name of steady state file to restart from"); CHKERRQ(ierr);
  
                                
  /* report contents of parameter structure */
  ierr = PetscPrintf(user.comm,"--------------------------------------\n"); CHKERRQ(ierr);
  ierr = PetscBagView(user.bag,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
  ierr = PetscPrintf(user.comm,"--------------------------------------\n"); CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
/* Create DMDA vectors and sets up SNES */
PetscErrorCode DMDASNESsetup(AppCtx *user)
/* ------------------------------------------------------------------- */
{
  PetscErrorCode ierr;
  PetscInt       dofs;
  PetscFunctionBegin;
  
  /* set up solution and residual vectors for the pressure */
  dofs = (PetscInt)(sizeof(FieldP)/sizeof(PetscReal));
  ierr = DMDACreate1d(user->comm,DM_BOUNDARY_NONE,user->param->ni,dofs,2,PETSC_NULL,&user->daP); CHKERRQ(ierr);
  ierr = DMSetUp(user->daP); CHKERRQ(ierr);
  ierr = DMDASetFieldName(user->daP,0,"P"); CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(user->daP,&user->XP); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)user->XP,"grid"); CHKERRQ(ierr);
  ierr = VecDuplicate(user->XP,&user->RP); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)user->RP,"res"); CHKERRQ(ierr);
  ierr = VecDuplicate(user->XP,&user->XPold); CHKERRQ(ierr);
  ierr = VecDuplicate(user->XP,&user->XPtheta); CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)user->XP,"xp_"); CHKERRQ(ierr); // unique prefix to vector that enables searching for its options
  ierr = PetscObjectSetOptionsPrefix((PetscObject)user->RP,"rp_"); CHKERRQ(ierr); // unique prefix to vector that enables searching for its options
  
  /* set up solution and residual vectors for the enthalpy and composition */
  dofs = (PetscInt)(sizeof(FieldH)/sizeof(PetscReal));
  ierr = DMDACreate1d(user->comm,DM_BOUNDARY_NONE,user->param->ni,dofs,2,PETSC_NULL,&user->daH); CHKERRQ(ierr);
  ierr = DMSetUp(user->daH); CHKERRQ(ierr);
  ierr = DMDASetFieldName(user->daH,0,"H"); CHKERRQ(ierr);
  ierr = DMDASetFieldName(user->daH,1,"c"); CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(user->daH,&user->XH); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)user->XH,"grid"); CHKERRQ(ierr);
  ierr = VecDuplicate(user->XH,&user->RH); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)user->RH,"res"); CHKERRQ(ierr);
  ierr = VecDuplicate(user->XH,&user->XHold); CHKERRQ(ierr);
  ierr = VecDuplicate(user->XH,&user->XHtheta); CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)user->XH,"xh_"); CHKERRQ(ierr); // unique prefix to vector that enables searching for its options
  ierr = PetscObjectSetOptionsPrefix((PetscObject)user->RH,"rh_"); CHKERRQ(ierr); // unique prefix to vector that enables searching for its options
  
  /* set up solution and residual vectors for the pipe */
  dofs = (PetscInt)(sizeof(FieldPipe)/sizeof(PetscReal));
  ierr = DMDACreate1d(user->comm,DM_BOUNDARY_NONE,user->param->ni,dofs,2,PETSC_NULL,&user->daPipe); CHKERRQ(ierr);
  ierr = DMSetUp(user->daPipe); CHKERRQ(ierr);
  ierr = DMDASetFieldName(user->daPipe,0,"qp"); CHKERRQ(ierr);
  ierr = DMDASetFieldName(user->daPipe,1,"Tp"); CHKERRQ(ierr);
  ierr = DMDASetFieldName(user->daPipe,2,"cp"); CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(user->daPipe,&user->XPipe); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)user->XPipe,"grid"); CHKERRQ(ierr);
  ierr = VecDuplicate(user->XPipe,&user->RPipe); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)user->RPipe,"res"); CHKERRQ(ierr);
  ierr = VecDuplicate(user->XPipe,&user->XPipeold); CHKERRQ(ierr);
  ierr = VecDuplicate(user->XPipe,&user->XPipetheta); CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)user->XPipe,"xpipe_"); CHKERRQ(ierr); // unique prefix to vector that enables searching for its options
  ierr = PetscObjectSetOptionsPrefix((PetscObject)user->RPipe,"rpipe_"); CHKERRQ(ierr); // unique prefix to vector that enables searching for its options
  
  /* set up vector of auxilliary variables */
  dofs = (PetscInt)(sizeof(AuxFieldP)/sizeof(PetscReal));
  ierr = DMDACreate1d(user->comm,DM_BOUNDARY_NONE,user->param->ni,dofs,2,PETSC_NULL,&user->daPaux); CHKERRQ(ierr);
  ierr = DMSetUp(user->daPaux); CHKERRQ(ierr);
  ierr = DMDASetFieldName(user->daPaux,0,"q"); CHKERRQ(ierr);
  ierr = DMDASetFieldName(user->daPaux,1,"u"); CHKERRQ(ierr);
  ierr = DMDASetFieldName(user->daPaux,2,"r"); CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(user->daPaux,&user->Paux); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)user->Paux,"grid"); CHKERRQ(ierr);
  ierr = VecDuplicate(user->Paux,&user->Pauxold); CHKERRQ(ierr);
  ierr = VecDuplicate(user->Paux,&user->Pauxtheta); CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)user->Paux,"paux_"); CHKERRQ(ierr); // unique prefix to vector that enables searching for its options
  ierr = PetscObjectSetOptionsPrefix((PetscObject)user->Paux,"paux_"); CHKERRQ(ierr); // unique prefix to vector that enables searching for its options

  /* set up vector of auxilliary variables */
  dofs = (PetscInt)(sizeof(AuxFieldH)/sizeof(PetscReal));
  ierr = DMDACreate1d(user->comm,DM_BOUNDARY_NONE,user->param->ni,dofs,2,PETSC_NULL,&user->daHaux); CHKERRQ(ierr);
  ierr = DMSetUp(user->daHaux); CHKERRQ(ierr);
  ierr = DMDASetFieldName(user->daHaux,0,"phi"); CHKERRQ(ierr);
  ierr = DMDASetFieldName(user->daHaux,1,"T"); CHKERRQ(ierr);
  ierr = DMDASetFieldName(user->daHaux,2,"cl"); CHKERRQ(ierr);
  ierr = DMDASetFieldName(user->daHaux,3,"cs"); CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(user->daHaux,&user->Haux); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)user->Haux,"grid"); CHKERRQ(ierr);
  ierr = VecDuplicate(user->Haux,&user->Hauxold); CHKERRQ(ierr);
  ierr = VecDuplicate(user->Haux,&user->Hauxtheta); CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)user->Haux,"haux_"); CHKERRQ(ierr); // unique prefix to vector that enables searching for its options
  ierr = PetscObjectSetOptionsPrefix((PetscObject)user->Haux,"haux_"); CHKERRQ(ierr); // unique prefix to vector that enables searching for its options

  /* set up vector of additional output variables */
  dofs = (PetscInt)(sizeof(FieldOUT)/sizeof(PetscReal));
  ierr = DMDACreate1d(user->comm,DM_BOUNDARY_NONE,user->param->ni,dofs,2,PETSC_NULL,&user->daOUT); CHKERRQ(ierr);
  ierr = DMSetUp(user->daOUT); CHKERRQ(ierr);
  ierr = DMDASetFieldName(user->daOUT,0,"Gamma"); CHKERRQ(ierr);
  ierr = DMDASetFieldName(user->daOUT,1,"Gamma_A"); CHKERRQ(ierr);
  ierr = DMDASetFieldName(user->daOUT,2,"E"); CHKERRQ(ierr);
  ierr = DMDASetFieldName(user->daOUT,3,"M"); CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(user->daOUT,&user->XOUT); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)user->XOUT,"grid"); CHKERRQ(ierr);
  
  /* set up nonlinear solver context for pressure equation */
  ierr = SNESCreate(user->comm,&user->snesP);CHKERRQ(ierr);
  ierr = SNESSetOptionsPrefix(user->snesP,"P_");CHKERRQ(ierr);
  ierr = SNESSetDM(user->snesP,user->daP);CHKERRQ(ierr);
  ierr = SNESSetFunction(user->snesP,user->RP,FormResidualP,user);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(user->snesP);CHKERRQ(ierr);
  
  /* set up nonlinear solver context for enthalpy composition equations */
  ierr = SNESCreate(user->comm,&user->snesH);CHKERRQ(ierr);
  ierr = SNESSetOptionsPrefix(user->snesH,"H_");CHKERRQ(ierr);
  ierr = SNESSetDM(user->snesH,user->daH);CHKERRQ(ierr);
  ierr = SNESSetFunction(user->snesH,user->RH,FormResidualH,user);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(user->snesH);CHKERRQ(ierr);
  
  /* set up nonlinear solver context for pipe */
  ierr = SNESCreate(user->comm,&user->snesPipe);CHKERRQ(ierr);
  ierr = SNESSetOptionsPrefix(user->snesPipe,"pipe_");CHKERRQ(ierr);
  ierr = SNESSetDM(user->snesPipe,user->daPipe);CHKERRQ(ierr);
  ierr = SNESSetFunction(user->snesPipe,user->RPipe,FormResidualPipe,user);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(user->snesPipe);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
/* Clean up by destroying vectors etc */
PetscErrorCode CleanUp(AppCtx *user)
/* ------------------------------------------------------------------- */
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  
  /* clean up by destroying objects that were created */
  ierr = VecDestroy(&user->XP);CHKERRQ(ierr);
  ierr = VecDestroy(&user->RP);CHKERRQ(ierr);
  ierr = VecDestroy(&user->XPold);CHKERRQ(ierr);
  ierr = VecDestroy(&user->XPtheta);CHKERRQ(ierr);
  ierr = VecDestroy(&user->XH);CHKERRQ(ierr);
  ierr = VecDestroy(&user->RH);CHKERRQ(ierr);
  ierr = VecDestroy(&user->XHold);CHKERRQ(ierr);
  ierr = VecDestroy(&user->XHtheta);CHKERRQ(ierr);
  ierr = VecDestroy(&user->XPipe);CHKERRQ(ierr);
  ierr = VecDestroy(&user->RPipe);CHKERRQ(ierr);
  ierr = VecDestroy(&user->XPipeold);CHKERRQ(ierr);
  ierr = VecDestroy(&user->XPipetheta);CHKERRQ(ierr);
  ierr = VecDestroy(&user->Haux);CHKERRQ(ierr);
  ierr = VecDestroy(&user->Paux);CHKERRQ(ierr);
  ierr = VecDestroy(&user->Hauxold);CHKERRQ(ierr);
  ierr = VecDestroy(&user->Pauxold);CHKERRQ(ierr);
  ierr = VecDestroy(&user->Hauxtheta);CHKERRQ(ierr);
  ierr = VecDestroy(&user->Pauxtheta);CHKERRQ(ierr);
  ierr = SNESDestroy(&user->snesP);CHKERRQ(ierr);
  ierr = SNESDestroy(&user->snesH);CHKERRQ(ierr);
  ierr = SNESDestroy(&user->snesPipe);CHKERRQ(ierr);
  ierr = DMDestroy(&user->daP);CHKERRQ(ierr);
  ierr = DMDestroy(&user->daH);CHKERRQ(ierr);
  ierr = DMDestroy(&user->daPipe);CHKERRQ(ierr);
  ierr = DMDestroy(&user->daHaux);CHKERRQ(ierr);
  ierr = DMDestroy(&user->daPaux);CHKERRQ(ierr);
  ierr = PetscBagDestroy(&user->bag);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return(ierr);
}

/* ------------------------------------------------------------------- */
PetscErrorCode projCreateDirectory(const char dirname[])
/* This generates a new directory called dirname. If dirname already exists,
   nothing happens. Importantly,
   a) only rank 0 tries to create the directory.
   b) nested directorys cannot be created, e.g. dirname[] = "output/step0/allmydata"
   is not valid. Instead you would have to call the function 3 times
   projCreateDirectory("output/");
   projCreateDirectory("output/step0");
   projCreateDirectory("output/step0/allmydata")
   ** Writen by Dave May, ask him if help needed **; */
/* ------------------------------------------------------------------- */
{
  PetscMPIInt rank;
  int num,error_number;
  PetscBool proj_log = PETSC_FALSE;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  
  /* Let rank 0 create a new directory on proc 0 */
  if (rank == 0) {
    num = mkdir(dirname,S_IRWXU);
    error_number = errno;
  }
  ierr = MPI_Bcast(&num,1,MPI_INT,0,PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = MPI_Bcast(&error_number,1,MPI_INT,0,PETSC_COMM_WORLD);CHKERRQ(ierr);
  
  if (error_number == EEXIST) {
    if (proj_log) PetscPrintf(PETSC_COMM_WORLD,"[proj] Writing output to existing directory: %s\n",dirname);
  } else if (error_number == EACCES) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"[proj] Write permission is denied for the parent directory in which the new directory is to be added");
  } else if (error_number == EMLINK) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"[proj] The parent directory has too many links (entries)");
  } else if (error_number == ENOSPC) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"[proj] The file system doesn't have enough room to create the new directory");
  } else if (error_number == ENOSPC) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"[proj] The parent directory of the directory being created is on a read-only file system and cannot be modified");
  } else {
    if (proj_log) PetscPrintf(PETSC_COMM_WORLD,"[proj] Created output directory: %s\n",dirname);
  }
  
  ierr = MPI_Barrier(PETSC_COMM_WORLD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
/* To make an output file */
PetscErrorCode DoOutput(AppCtx *user, int it, PetscBool STEADY)
/* ------------------------------------------------------------------- */
{
  Parameter      *p = user->param;
  char*          filename = NULL;
  PetscInt       hhat_A = p->hhat_A, hhat_B = p->hhat_B, c = p->bulk_comp*100;
  PetscInt       Pc = p->Pc*10, Te = p->Te*10;
  PetscViewer    viewer;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  
  if (!STEADY) {
    asprintf(&filename,"%s%s_%04d",p->output_dir,p->filename,it);
  } else {
    asprintf(&filename,"%sSSc%03d_ha%03d_hb%03d_Pc%02d_Te%02d",p->SS_output_dir,c,hhat_A,hhat_B,Pc,Te);
  }
  
  if (p->it == p->ns) {
    asprintf(&filename,"%sNOSSc%03d_ha%03d_hb%03d_Pc%02d_Te%02d",p->SS_output_dir,c,hhat_A,hhat_B,Pc,Te);
  }
  
  ierr = Extra_output_params(user);CHKERRQ(ierr); // calculate extra outputting parameters
  
  ierr = PetscPrintf(user->comm," Generating output file: \"%s\"\n",filename);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(user->comm,filename,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_BINARY_MATLAB);CHKERRQ(ierr);
  ierr = PetscBagView(user->bag,viewer);CHKERRQ(ierr); // output bag
  ierr = VecView(user->XP,viewer);CHKERRQ(ierr);        // output solution
  ierr = VecView(user->XH,viewer);CHKERRQ(ierr);        // output solution
  ierr = VecView(user->XPipe,viewer);CHKERRQ(ierr);        // output solution
  ierr = VecView(user->XOUT,viewer);CHKERRQ(ierr);        // output solution
  ierr = VecView(user->Haux,viewer);CHKERRQ(ierr);     // output auxilliary/diagnostic variables
  ierr = VecView(user->Paux,viewer);CHKERRQ(ierr);     // output auxilliary/diagnostic variables
  ierr = VecView(user->RP,viewer);CHKERRQ(ierr);        // output residual
  ierr = VecView(user->RH,viewer);CHKERRQ(ierr);        // output residual
  ierr = VecView(user->RPipe,viewer);CHKERRQ(ierr);        // output residual
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  free(filename);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
/* Import state from an output file */
PetscErrorCode RestartFromFile(AppCtx *user)
/* ------------------------------------------------------------------- */
{
  Parameter      *p = user->param;
  char*          filename = NULL;
  PetscViewer    viewer;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  
  if (p->restart_step == -1) {
    asprintf(&filename,"%s%s",p->SS_output_dir,p->restart_ss);
  } else {
    asprintf(&filename,"%s%s_%04d",p->output_dir,p->filename,p->restart_step);
  }
  
  ierr = PetscPrintf(user->comm," Restarting from file: \"%s\"\n",filename);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(user->comm,filename,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = PetscBagLoad(viewer,user->bag);CHKERRQ(ierr); // read in bag
  ierr = VecLoad(user->XP,viewer);CHKERRQ(ierr);  // read in solution
  ierr = VecLoad(user->XH,viewer);CHKERRQ(ierr);  // read in solution
  ierr = VecLoad(user->XPipe,viewer);CHKERRQ(ierr); // read in solution
  ierr = VecLoad(user->XOUT,viewer);CHKERRQ(ierr); // read in solution
  ierr = VecLoad(user->Haux,viewer);CHKERRQ(ierr);  // read in auxilliary/diagnostic variables
  ierr = VecLoad(user->Paux,viewer);CHKERRQ(ierr);  // read in auxilliary/diagnostic variables
  ierr = VecLoad(user->RP,viewer);CHKERRQ(ierr);  // read in residual
  ierr = VecLoad(user->RH,viewer);CHKERRQ(ierr);  // read in residual
  ierr = VecLoad(user->RPipe,viewer);CHKERRQ(ierr);  // read in residual
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  free(filename);
  
  ierr = PetscBagSetFromOptions(user->bag);CHKERRQ(ierr); // take any new options from command line
  
  PetscPrintf(user->comm,"it = %i \n",p->it);
  
  /* Re-calculate reference values and non-dim values */
  p->psi_0 = p->Psi_0/(4.0/3.0 * PETSC_PI * (pow(p->R,3)-pow(p->R_cmb,3))); // set reference constant heating rate in W/m3
  p->Pe = p->psi_0*p->R*p->R/(p->L*p->kappa*p->rho_0); // Peclet number
  p->Pec = p->psi_0*p->R*p->R/(p->L*p->D*p->rho_0); // compositional Peclet number
  p->St = p->L/(p->ce*p->T_0); // Stefan number
  p->phi_0 = pow(p->psi_0*p->R*p->eta_l/(p->L*p->rho_0*p->K_0*p->del_rho*p->g),1/p->perm); // reference porosity
  p->zeta_0 = p->eta/p->phi_0;
  p->delta = p->zeta_0*p->K_0*pow(p->phi_0,p->perm)/(p->eta_l*p->R*p->R);
  p->P_0 = p->zeta_0*p->psi_0/(p->rho_0*p->L); // reference pressure
  
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
/* Get useful information from the DMDA */
PetscErrorCode DMDAGetGridInfo(DM da, int *is, int *js, int *ks, int *ie,
             int *je, int *ke, int *ni, int *nj, int *nk,
             int *dim)
/* ------------------------------------------------------------------- */
{
  PetscErrorCode ierr;
  PetscInt       im, jm, km;
  PetscFunctionBegin;
  ierr = DMDAGetCorners(da,is,js,ks,&im,&jm,&km); CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,dim,ni,nj,nk,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  if (ie) *ie = *is + im;
  if (je) *je = *js + jm;
  if (ke) *ke = *ks + km;
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
/* Gets array from a vector associated with a DMDA, with ghost points */
PetscErrorCode DAGetGhostedArray(DM da, Vec globvec, Vec *locvec, void *arr)
/* ------------------------------------------------------------------- */
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = DMGetLocalVector(da,locvec);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da,globvec,INSERT_VALUES,*locvec);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd  (da,globvec,INSERT_VALUES,*locvec);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,*locvec,arr); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
/* Restores array from a vector associated with a DMDA, with ghost points */
PetscErrorCode DARestoreGhostedArray(DM da, Vec globvec, Vec *locvec, void *arr)
/* ------------------------------------------------------------------- */
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = DMDAVecRestoreArray(da,*locvec,arr); CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(da,*locvec,INSERT_VALUES,globvec);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (da,*locvec,INSERT_VALUES,globvec);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,locvec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
