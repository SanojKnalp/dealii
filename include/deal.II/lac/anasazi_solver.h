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


#ifndef dealii_anasazi_trilinos_solver
#define dealii_anasazi_trilinos_solver


#include <deal.II/base/config.h>
#ifdef DEAL_II_TRILINOS_WITH_ANASAZI
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/trilinos_tpetra_vector.h>
#include <deal.II/lac/trilinos_tpetra_sparse_matrix.h>

#include <AnasaziBasicEigenproblem.hpp>
#include <AnasaziTpetraAdapter.hpp>
#endif

DEAL_II_NAMESPACE_OPEN

#ifdef DEAL_II_TRILINOS_WITH_ANASAZI

/**
 * To do:
 * 1. Finish the remaining functions that are not implemented yet
 * 2. See if we can also add the other SLEPc functions
 * 3. Make sure it works with Tpetra and Epetra
 * 4. Write tests by transforming the SLEPc tests into Anasazi tests 
 * 5. Add more Solvers
 * 6. Add tests for those solvers
 */
namespace TrilinosWrappers
{
    class SolverBase
    {
        public:

        /**
         * Constructor. Takes in the SolverControl
         */
        SolverBase(SolverControl &cn,const AdditionalData &data=AdditionalData());

        /**
         * Destructor.
         */
        virtual ~SolverBase();

        template <typename InputMatrix, typename OutputVector>
        void
        solve(const InputMatrix &A,
             std::vector<TrilinosScalar>& eigenvalues,
            std::vector<OutputVector>& eigenvectors,
            const unsigned int	n_eigenpairs = 1 ); //implemented

        template <typename OutputVector>
        void
        solve(const InputMatrix &A,
              const InputMatrix &B,
                std::vector<TrilinosScalar>& eigenvalues,
                std::vector<OutputVector>& eigenvectors,
                const unsigned int	n_eigenpairs = 1 ); //implemented

        template <typename OutputVector>
        void 
        solve(const InputMatrix &A,
              const InputMatrix &B,
              std::vector<double>&              real_eigenvalues,
              std::vector<double>&              imag_eigenvalues,
              std::vector<OutputVector>&        real_eigenvectors,
              std::vector<OutputVector>&        imag_eigenvectors,
              const unsigned int n_eigenpairs = 1); //implemented

        template <typename Vector>
        void
        set_initial_space(const std::vector<Vector>& initial_space);

        /**
         * Exception. Standard exception.
         */
        DeclException0(ExcAnasaziWrappersUsageError);

        /**
         * Exception. Convergence failure on the number of eigenvectors.
         */
        DeclException2(ExcAnasaziigenvectorConvergenceMismatchError,
                   int,
                   int,
                   << "    The number of converged eigenvectors is " << arg1
                   << " but " << arg2 << " were requested. ");

        void
        get_solver_state(const SolverControl::State state); //implemented

        /**
         * Access to the object that controls convergence.
         */
        SolverControl &
        control() const;

        protected:
        /**
         * Reference to the object that controls convergence of the iterative
         * solver.
         */
        SolverControl &solver_control;

        /**
         * Solve the linear system for <code>n_eigenpairs</code> eigenstates.
         * Parameter <code>n_converged</code> contains the actual number of
         * eigenstates that have  converged; this can be both fewer or more than
         * n_eigenpairs, depending on the Anasazi eigensolver used.
         */
        void
        solve(const unsigned int n_eigenpairs, unsigned int *n_converged); //started

        template <typename Vector>
        void
        get_eigenpair(const unsigned int index,
                        TrilinosScalar& eigenvalues,
                        Vector &eigenvector); 

        template <typename Vector>
        void
        get_eigenpair(const unsigned int index, 
                      double& real_eigenvalues,
                      double& imag_eigenvalues,
                      Vector& real_eigenvector,
                      Vector& imag_eigenvalues);

        template <typename Matrix>
        void 
        set_matrices(const Matrix& A); //implemented

        template <typename Matrix>
        void
        set_matrices(const Matrix& A, const Matrix& B); //implemented

        void
        set_solver_manager() = 0;

        protected:
        Anasazi::BasicEigenproblem eigenproblem;
        Anasazi::Eigensolution eigensolution;

        std::unique_ptr<AnasaziSolverBase> solver;

    };

    class SolverBlockKrylovSchur : public SolverBase
    {
        struct AdditionalData
        {};

        explicit SolverBlockKrylovSchur(SolverControl& cn, const AdditionalData &data = AdditionalData());

        protected:
        const AdditionalData additional_data;
    };

    // --------------------------- inline and template functions -----------
    template <typename InputMatrix, typename OutputVector>
    void 
    SolverBase::solve(const InputMatrix &A,
             std::vector<TrilinosScalar>& eigenvalues,
            std::vector<OutputVector>& eigenvectors,
            const unsigned int	n_eigenpairs)
    {
        // Panic if the number of eigenpairs wanted is out of bounds.
        AssertThrow((n_eigenpairs > 0) && (n_eigenpairs <= A.m()),
                ExcAnasaziWrappersUsageError());
                
        //set the matrix of the problem
        set_matrices(A);

        // and solve
        unsigned int n_converged = 0;
        solve(n_eigenpairs, &n_converged);

        if(n_converged > n_eigenpairs)
        {
            n_converged = n_eigenpairs;
        }
            
        AssertThrow(n_converged == n_eigenpairs,
                    ExcAnasaziEigenvectorConvergenceMismatchError(n_converged,
                                                                n_eigenpairs));

        AssertThrow(eigenvectors.size() != 0, ExcAnasaziWrappersUsageError());
        eigenvectors.resize(n_converged, eigenvectors.front());
        eigenvalues.resize(n_converged);

        for(unsigned int index = 0; index < n_converged; ++index)
        {
            get_eigenpair(index, eigenvalues[index], eigenvectors[index]);
        }
            
    }

    template <typename InputMatrix, typename OutputVector>
    void 
    SolverBase::solve((const InputMatrix &A,
              const InputMatrix &B,
                std::vector<TrilinosScalar>& eigenvalues,
                std::vector<OutputVector>& eigenvectors,
                const unsigned int	n_eigenpairs )
    {
        // Guard against incompatible matrix sizes:
        AssertThrow(A.m() == B.m(), ExcDimensionMismatch(A.m(), B.m()));
        AssertThrow(A.n() == B.n(), ExcDimensionMismatch(A.n(), B.n()));

         // Panic if the number of eigenpairs wanted is out of bounds.
        AssertThrow((n_eigenpairs > 0) && (n_eigenpairs <= A.m()),
                    ExcAnasaziWrappersUsageError());

        //set the matrix of the problem
        set_matrices(A, B);

        // and solve
        unsigned int n_converged = 0;
        solve(n_eigenpairs, &n_converged);

        if (n_converged > n_eigenpairs)
        n_converged = n_eigenpairs;
        AssertThrow(n_converged == n_eigenpairs,
                    ExcAnasaziEigenvectorConvergenceMismatchError(n_converged,
                                                                n_eigenpairs));

        AssertThrow(eigenvectors.size() != 0, ExcAnasaziWrappersUsageError());
        eigenvectors.resize(n_converged, eigenvectors.front());
        eigenvalues.resize(n_converged);

        for (unsigned int index = 0; index < n_converged; ++index)
        {
            get_eigenpair(index, eigenvalues[index], eigenvectors[index]);
        }
    }

    template <typename InputMatrix, typename OutputVector>
    void 
    SolverBase::solve(const InputMatrix &A,
              const InputMatrix &B,
              std::vector<double>&              real_eigenvalues,
              std::vector<double>&              imag_eigenvalues,
              std::vector<OutputVector>&        real_eigenvectors,
              std::vector<OutputVector>&        imag_eigenvectors,
              const unsigned int n_eigenpairs )
    {
        // Guard against incompatible matrix sizes:
        AssertThrow(A.m() == B.m(), ExcDimensionMismatch(A.m(), B.m()));
        AssertThrow(A.n() == B.n(), ExcDimensionMismatch(A.n(), B.n()));

        // and incompatible eigenvalue/eigenvector sizes
        AssertThrow(real_eigenvalues.size() == imag_eigenvalues.size(),
                    ExcDimensionMismatch(real_eigenvalues.size(),
                                        imag_eigenvalues.size()));
        AssertThrow(real_eigenvectors.size() == imag_eigenvectors.size(),
                    ExcDimensionMismatch(real_eigenvectors.size(),
                                        imag_eigenvectors.size()));

         // Panic if the number of eigenpairs wanted is out of bounds.
        AssertThrow((n_eigenpairs > 0) && (n_eigenpairs <= A.m()),
                    ExcAnasaziWrappersUsageError());

        //set the matrix of the problem
        set_matrices(A, B);

        // and solve
        unsigned int n_converged = 0;
        solve(n_eigenpairs, &n_converged);

        if (n_converged > n_eigenpairs)
        n_converged = n_eigenpairs;
        AssertThrow(n_converged == n_eigenpairs,
                    ExcAnasaziEigenvectorConvergenceMismatchError(n_converged,
                                                                n_eigenpairs));

        AssertThrow((real_eigenvectors.size() != 0) &&
                  (imag_eigenvectors.size() != 0),
                ExcAnasaziWrappersUsageError());
         real_eigenvectors.resize(n_converged, real_eigenvectors.front());
        imag_eigenvectors.resize(n_converged, imag_eigenvectors.front());
        real_eigenvalues.resize(n_converged);
        imag_eigenvalues.resize(n_converged);

        for (unsigned int index = 0; index < n_converged; ++index)
        {
            get_eigenpair(index,
                            real_eigenvalues[index],
                            imag_eigenvalues[index],
                            real_eigenvectors[index],
                            imag_eigenvectors[index]);

        }   
    }

    template <Matrix>
    void 
    SolverBase::set_matrices(const Matrix& A)
    {
        eigenproblem.setA(A);
    }

    template <Matrix>
    void SolverBase::set_matrices(const Matrix& A, const Matrix& B)
    {
        set_matrices(A);
        eigenproblem.setM(B);
    }


        
}

#endif

DEAL_II_NAMESPACE_CLOSE

#endif