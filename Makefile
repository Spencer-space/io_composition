CFLAGS   = 
FFLAGS   = 
CPPFLAGS =
FPPFLAGS =
PROG1    = iocomp
OBJECTS1 = ${PROG1}.o 

all:	${PROG1}

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

${PROG1}: ${OBJECTS1} chkopts
	${RM} ${PROG1}
	-${CLINKER} -o ${PROG1} ${OBJECTS1} ${PETSC_SNES_LIB}