// -----------------------------------------------------------------------------
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later
// Copyright (C) 2009 - 2026 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Detailed license information governing the source code and contributions
// can be found in LICENSE.md and CONTRIBUTING.md at the top level directory.
//
// -----------------------------------------------------------------------------

#include <deal.II/lac/anasazi_solver.h>

DEAL_II_NAMESPACE_OPEN

#ifdef DEAL_II_TRILINOS_WITH_ANASAZI

namespace TrilinosWrappers
{
    SolverBase::SolverBase(SolverControl &cn,const AdditionalData &data)
    : solver_control(cn)
    , additional_data(data)
    {
        set_solver_type();
    }

    SolverBase::~SolverBase()
    {
        eigenproblem.~BasicEigenproblem();
    }

    void
    SolverBase::solve(const unsigned int n_eigenpairs, unsigned int *n_converged)
    {
        // set the number of eigenvalues to be computed
        eigenproblem.NEV(static_cast<int>(n_eigenpairs));

        // basically we tell Aasazi that we are ready
        eigenproblem.setProblem();

        solver->solve();

        eigensolution = eigenproblem.getSolution();

        n_converged = static_cast<unsigned int>(eigensolution.numVecs);
    }
}

#endif

DEAL_II_NAMESPACE_CLOSE