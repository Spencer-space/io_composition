#include "petsc.h"

/* this is a MACRO, a find/replace item for the preprocessor */
#define FNAME_LENGTH  120

/* Defining a macro, changing the console text colour */
#define CHANGE_CONSOLE_COLOUR(comm, colour) PetscPrintf(comm, colour)
#define BOLD_RED "\033[1;31m"
#define BOLD_GREEN "\033[1;32m"
#define BOLD_YELLOW "\033[1;33m"
#define BOLD_BLUE "\033[1;34m"
#define COLOUR_RESET "\033[0m"

/* solution structure, containing the set of variables (degrees of freedom)
   that we wish to solve for *at each point in the domain* */
typedef struct  {
  PetscReal P;
} FieldP;

typedef struct  {
  PetscReal H;
  PetscReal c;
} FieldH;

typedef struct  {
  PetscReal qp;
  PetscReal Tp;
  PetscReal cp;
} FieldPipe;

typedef struct {
  PetscReal q;
  PetscReal u;
  PetscReal r;
} AuxFieldP;

typedef struct {
  PetscReal phi;
  PetscReal T;
  PetscReal cl;
  PetscReal cs;
} AuxFieldH;

typedef struct {
  PetscReal Gamma;
  PetscReal Gamma_A;
  PetscReal E;
  PetscReal M;
} FieldOUT;

/* parameter structure, containing the problem parameters */
typedef struct {
  PetscInt  ni, ns;   //number of grid points, no. timesteps
  PetscReal R_cmb, R, base, g;   // Io structure parameters
  PetscReal perm, L, ce, K_0, eta, eta_l, rho_0, del_rho, kappa, zeta_0, D;   // Io material parameters
  PetscReal Pe, Pec, St, delta, phi_0, hhat_A, hhat_B, nu, delta_t, delta_phi, delta_xi, delta_alpha, Pc, P_0; // Io non-dimensional parameters and eruption constants
  PetscReal T_0, T_B, T_A, T_surf_dim, Te;   // temperature and composition parameters
  PetscReal psi_0, Psi_0, psi;
  PetscReal H_end;   // boundary condition parameters
  PetscReal gamma, bulk_comp; // composition parameters
  PetscReal t, CFL, dt, tmax, theta, steady_tol, dtau;   // timestepping parameters
  PetscInt  it, SNESit, true_solve_flag;   // timestepping integer parameters
  PetscInt  out_freq;   // outputting parameters
  PetscInt  restart_step; // restart parameters
  char      restart_ss[PETSC_MAX_PATH_LEN]; // when a steady state file is used to restart
  char      output_dir[PETSC_MAX_PATH_LEN]; // folder where output files will go
  char      SS_output_dir[PETSC_MAX_PATH_LEN]; // folder where output files will go
  char      filename[FNAME_LENGTH];
} Parameter;

/* application context structure, containing the main data structures needed
   for the simulation */
typedef struct {
  Parameter     *param;
  PetscBag      bag;
  DM            daP, daH, daPipe, daHaux, daPaux, daOUT;
  SNES          snesP, snesH, snesPipe;
  SNESConvergedReason reasonP, reasonH, reasonPipe, reasonPipePT;
  Vec           XP, RP, XPold, Haux, Paux, Hauxold, Pauxold, Hauxtheta, Pauxtheta;
  Vec           XH, RH, XHold, XPtheta, XHtheta;
  Vec           XPipe, RPipe, XPipeold, XPipetheta;
  Vec           XOUT; // a few extra parameters to calculate and output
  MPI_Comm      comm;
} AppCtx;

/* function declarations */
PetscErrorCode  Initialisation(AppCtx*, Vec, Vec, Vec, Vec, Vec);
PetscErrorCode  TimeStepping(AppCtx*);
PetscErrorCode  PseudoTransient(AppCtx*, PetscReal);
PetscErrorCode  FormResidualP(SNES, Vec, Vec, void*);
PetscErrorCode  FormResidualH(SNES, Vec, Vec, void*);
PetscErrorCode  FormResidualPipe(SNES, Vec, Vec, void*);
PetscErrorCode  FormResidualPipePT(SNES, Vec, Vec, void*);
PetscReal PressureResidual(AppCtx*, const FieldP*, FieldPipe*, AuxFieldH*, AuxFieldP*, PetscInt);
PetscReal EnthalpyResidual(AppCtx*, const FieldH*, FieldH*, FieldH*, FieldP*, FieldPipe*, AuxFieldH*, AuxFieldP*, PetscInt i);
PetscReal CompositionResidual(AppCtx*, const FieldH*, FieldH*, FieldH*, FieldP*, FieldPipe*, AuxFieldH*, AuxFieldP*, PetscInt i);
PetscReal qpResidual(AppCtx*, const FieldPipe*, FieldP*, AuxFieldH*, AuxFieldP*, PetscInt);
PetscReal TpResidual(AppCtx*, const FieldPipe*, FieldP*, AuxFieldH*, AuxFieldP*, PetscInt);
PetscReal cpResidual(AppCtx*, const FieldPipe*, FieldP*, AuxFieldH*, AuxFieldP*, PetscInt);
PetscErrorCode  PAuxParamsCalc(AppCtx*, const FieldP*, FieldPipe*, AuxFieldH*, AuxFieldP*);
PetscErrorCode  HAuxParamsCalc(AppCtx*, const FieldH*, AuxFieldH*);
PetscBool SteadyStateChecker(AppCtx*);
PetscReal  Fromm_advection(AppCtx*, PetscReal, PetscReal, PetscReal, PetscReal, PetscReal, PetscReal, PetscReal, PetscReal);
PetscErrorCode  Extra_output_params(AppCtx*);
PetscErrorCode  ParameterSetup(AppCtx);
PetscErrorCode  DMDASNESsetup(AppCtx*);
PetscErrorCode  CleanUp(AppCtx*);
PetscErrorCode  projCreateDirectory(const char dirname[]);
PetscErrorCode  DoOutput(AppCtx*, int, PetscBool);
PetscErrorCode  RestartFromFile(AppCtx*);
PetscErrorCode  DAGetGhostedArray(DM, Vec, Vec*, void*);
PetscErrorCode  DARestoreGhostedArray(DM, Vec, Vec*, void*);
PetscErrorCode  DMDAGetGridInfo(DM, int*, int*, int*, int*, int*, int*, int*, int*, int*, int*);

