/**
 * Standalone correctness test for DistributeMatrixFromRank0() and
 * DistributeVectorFromRank0() (see Distribute.hpp).
 *
 * The matrix/vector for step 1 are loaded two different ways and the
 * results compared:
 *
 *   A_glb, b_glb  -- loaded directly on PETSC_COMM_WORLD. MatLoad/VecLoad
 *                    split the on-disk (serial) data among ranks according
 *                    to PETSc's own row decomposition. This is the
 *                    reference layout/values.
 *
 *   A_seq, b_seq  -- loaded whole on rank 0 only (PETSC_COMM_SELF), then
 *                    handed to DistributeMatrixFromRank0/
 *                    DistributeVectorFromRank0 to produce A, b.
 *
 * If the two distribution helpers are correct, A == A_glb and b == b_glb
 * (up to a strict tolerance -- both paths read the same binary file, so
 * any mismatch is a bug in the distribution code, not numerical noise).
 *
 * Does not touch Main.cpp: this is a separate executable (see
 * iface/CMakeLists.txt, target testdistribute) that only exercises the
 * two helpers declared in Distribute.hpp.
 */

#include <iostream>
#include <string>
#include <unistd.h>

#include "DREAM/Init.h"
#include "Distribute.hpp"

using namespace std;

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int my_rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    PETSC_COMM_WORLD = MPI_COMM_WORLD;
    dream_initialize();

    const int step = 1;
    string matname = "petsc_mat_serial_step" + to_string(step) + "_iter1";
    string rhsname = "petsc_rhs_serial_step" + to_string(step) + "_iter1";

    int haveStep = 0;
    if (my_rank == 0)
        haveStep = (access(matname.c_str(), F_OK) == 0 &&
                    access(rhsname.c_str(), F_OK) == 0);
    MPI_Bcast(&haveStep, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!haveStep)
    {
        if (my_rank == 0)
            cerr << "Could not find " << matname << " / " << rhsname
                 << " in the current directory -- run the serial simulation "
                    "first to produce them."
                 << endl;
        dream_finalize();
        MPI_Finalize();
        return 1;
    }

    PetscViewer viewer;

    // ---- reference: load directly in parallel on PETSC_COMM_WORLD ----
    Mat A_glb;
    Vec b_glb;

    PetscViewerBinaryOpen(PETSC_COMM_WORLD, matname.c_str(), FILE_MODE_READ, &viewer);
    MatCreate(PETSC_COMM_WORLD, &A_glb);
    MatSetType(A_glb, MATAIJ);
    MatLoad(A_glb, viewer);
    PetscViewerDestroy(&viewer);

    PetscViewerBinaryOpen(PETSC_COMM_WORLD, rhsname.c_str(), FILE_MODE_READ, &viewer);
    VecCreate(PETSC_COMM_WORLD, &b_glb);
    VecLoad(b_glb, viewer);
    PetscViewerDestroy(&viewer);

    // ---- test path: load whole on rank 0, then distribute ----
    // Timed separately: tLoad is rank 0's serial MatLoad/VecLoad (only rank
    // 0 does any work, but the barriers keep the timing window the same on
    // every rank); tSend is the collective distribute calls, i.e. rank 0
    // packing and sending its data to every other rank.
    Mat A_seq = NULL;
    Vec b_seq = NULL;

    MPI_Barrier(PETSC_COMM_WORLD);
    double tLoad0 = MPI_Wtime();
    if (my_rank == 0)
    {
        PetscViewerBinaryOpen(PETSC_COMM_SELF, matname.c_str(), FILE_MODE_READ, &viewer);
        MatCreate(PETSC_COMM_SELF, &A_seq);
        MatSetType(A_seq, MATSEQAIJ);
        MatLoad(A_seq, viewer);
        PetscViewerDestroy(&viewer);

        PetscViewerBinaryOpen(PETSC_COMM_SELF, rhsname.c_str(), FILE_MODE_READ, &viewer);
        VecCreate(PETSC_COMM_SELF, &b_seq);
        VecLoad(b_seq, viewer);
        PetscViewerDestroy(&viewer);
    }
    MPI_Barrier(PETSC_COMM_WORLD);
    double tLoad = MPI_Wtime() - tLoad0;

    Mat A = NULL;
    Vec b = NULL, x = NULL;

    MPI_Barrier(PETSC_COMM_WORLD);
    double tSend0 = MPI_Wtime();

    DistributeMatrixFromRank0(A_seq, &A); // collective
    if (my_rank == 0)
        MatDestroy(&A_seq);

    MatCreateVecs(A, &x, nullptr); // x gets A's row layout, for use as the
                                    // DistributeVectorFromRank0 layout template
    DistributeVectorFromRank0(x, b_seq, &b); // collective
    if (my_rank == 0)
        VecDestroy(&b_seq);

    MPI_Barrier(PETSC_COMM_WORLD);
    double tSend = MPI_Wtime() - tSend0;

    if (my_rank == 0)
        cout << "[timing] rank-0 MatLoad+VecLoad: " << tLoad << " s"
             << "   distribute to all ranks: " << tSend << " s" << endl;

    // ---- layout sanity print ----
    PetscInt row_start, row_end, M, N, M_glb, N_glb;
    MatGetOwnershipRange(A, &row_start, &row_end);
    MatGetSize(A, &M, &N);
    MatGetSize(A_glb, &M_glb, &N_glb);
    PetscSynchronizedPrintf(PETSC_COMM_WORLD,
                            "[rank %d] A local rows [%d, %d)\n",
                            my_rank, (int)row_start, (int)row_end);
    PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT);
    if (my_rank == 0)
        cout << "[matrix] A size " << M << " x " << N
             << "   A_glb size " << M_glb << " x " << N_glb << endl;

    // ---- compare A vs A_glb ----
    {
        Mat diff;
        MatDuplicate(A, MAT_COPY_VALUES, &diff);
        MatAXPY(diff, -1.0, A_glb, DIFFERENT_NONZERO_PATTERN);

        PetscReal diffNorm, refNorm;
        MatNorm(diff, NORM_FROBENIUS, &diffNorm);
        MatNorm(A_glb, NORM_FROBENIUS, &refNorm);
        PetscReal relError = (refNorm > 0.0) ? diffNorm / refNorm : diffNorm;

        const double tol = 1e-12;
        if (my_rank == 0)
            cout << "matrix: ||A - A_glb||_F / ||A_glb||_F = " << relError
                 << (relError <= tol ? "  (MATCH)" : "  (MISMATCH)") << endl;

        MatDestroy(&diff);
    }

    // ---- compare b vs b_glb ----
    {
        Vec diff;
        VecDuplicate(b, &diff);
        VecWAXPY(diff, -1.0, b_glb, b);

        PetscReal diffNorm, refNorm;
        VecNorm(diff, NORM_2, &diffNorm);
        VecNorm(b_glb, NORM_2, &refNorm);
        PetscReal relError = (refNorm > 0.0) ? diffNorm / refNorm : diffNorm;

        const double tol = 1e-12;
        if (my_rank == 0)
            cout << "vector: ||b - b_glb|| / ||b_glb|| = " << relError
                 << (relError <= tol ? "  (MATCH)" : "  (MISMATCH)") << endl;

        VecDestroy(&diff);
    }

    MatDestroy(&A);
    MatDestroy(&A_glb);
    VecDestroy(&b);
    VecDestroy(&x);
    VecDestroy(&b_glb);

    dream_finalize();
    MPI_Finalize();
    return 0;
}
