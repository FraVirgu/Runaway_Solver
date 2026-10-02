/**
 * Non-linear solver for the EqsysInitializer.
 */

#include "DREAM/EqsysInitializer.hpp"
#include "DREAM/Solver/SolverNonLinear.hpp"

using namespace std;
using namespace DREAM;

/**
 * Solve a non-linear system of steady-state equations for
 * the specified unknown quantities.
 */
void EqsysInitializer::NonLinearSolve(const real_t t0, vector<len_t> &ssQty)
{

	SolverNonLinear *snl = new SolverNonLinear(
		this->unknowns, this->unknown_equations, this->eqsys,
		this->linear_solver, this->backup_solver,
		this->solver_maxiter, this->solver_reltol, this->solver_verbose);

	snl->SetCollisionHandlers(
		eqsys->GetHotTailCollisionHandler(),
		eqsys->GetRunawayCollisionHandler(),
		eqsys->GetREFluid());
	snl->SetSPIHandler(eqsys->GetSPIHandler());
	snl->SetIonHandler(eqsys->GetIonHandler());
	snl->SetBootstrap(eqsys->GetBootstrap());

	// Calculate number of rows in matrix...
	len_t matrix_size = 0;
	for (int_t id : ssQty)
		matrix_size += this->unknowns->GetUnknown(id)->NumberOfElements();
	std::cout << "-------------" << std::endl;
	snl->Initialize(matrix_size, ssQty);

	// Specify initial guess of solution
	const real_t *guess = this->unknowns->GetLongVector(ssQty);
	std::cout << "before  SetInitialGuess t=t0 non linear Solver" << std::endl;
	snl->SetInitialGuess(guess);
	std::cout << "after  SetInitialGuess t=t0 non linear Solver" << std::endl;
	snl->Solve(t0, 0);
	std::cout << "after  Solve t=t0 non linear Solver" << std::endl;
	for (int_t id : ssQty)
	{
		std::cout << "-1111111111111111111111111111111111111-" << std::endl;
		const real_t *data = this->unknowns->GetUnknownData(id);
		this->unknowns->SetInitialValue(id, data, t0);
	}

	delete snl;
	delete[] guess;
}
