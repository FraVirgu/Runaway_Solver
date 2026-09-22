#ifndef _RUNAWAY_SOLVER_IFACE_DISTRIBUTE_HPP
#define _RUNAWAY_SOLVER_IFACE_DISTRIBUTE_HPP

/**
 * Helpers to distribute a matrix/vector that lives entirely on rank 0 into
 * a matrix/vector split across all ranks of PETSC_COMM_WORLD, matching
 * whatever row layout PETSc decides (or has already decided, for the
 * vector case).
 *
 * Defined inline because this header is included from more than one
 * translation unit (Main.cpp and any test executable); a plain non-inline
 * definition here would cause multiple-definition link errors.
 */

#include <vector>
#include <mpi.h>
#include <petscmat.h>

/**
 * Distribute a SeqAIJ (or otherwise rank-0-resident) matrix Source into a
 * freshly created MPIAIJ matrix *A, split across PETSC_COMM_WORLD according
 * to PETSc's own row decomposition. Collective: every rank must call this.
 *
 * Source is only read on rank 0 (it may be left unset/NULL elsewhere).
 * On return, *A is fully assembled and ready to use.
 */
inline void DistributeMatrixFromRank0(Mat Source, Mat *A)
{
    PetscInt M = 0, N = 0;
    int my_rank, size;
    MPI_Comm_rank(PETSC_COMM_WORLD, &my_rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &size);

    if (my_rank == 0)
        MatGetSize(Source, &M, &N);
    MPI_Bcast(&M, 1, MPIU_INT, 0, PETSC_COMM_WORLD);
    MPI_Bcast(&N, 1, MPIU_INT, 0, PETSC_COMM_WORLD);

    // ---- create the destination matrix's layout ----
    MatCreate(PETSC_COMM_WORLD, A);
    MatSetSizes(*A, PETSC_DECIDE, PETSC_DECIDE, M, N);
    MatSetFromOptions(*A);
    MatSetUp(*A); // allocate with a generic (unoptimized) preallocation

    PetscInt Istart, Iend;
    MatGetOwnershipRange(*A, &Istart, &Iend);

    // ---- gather every rank's row range ----
    std::vector<PetscInt> core_start(size), core_end(size);
    MPI_Allgather(&Istart, 1, MPIU_INT, core_start.data(), 1, MPIU_INT, PETSC_COMM_WORLD);
    MPI_Allgather(&Iend, 1, MPIU_INT, core_end.data(), 1, MPIU_INT, PETSC_COMM_WORLD);

    if (my_rank == 0)
    {
        std::vector<MPI_Request> requests;

        // Per destination rank r, pack rows [core_start[r], core_end[r]) as CSR:
        // rowptr (num_rows+1), cols (nnz), vals (nnz). Kept alive until Waitall.
        std::vector<std::vector<PetscInt>> rowptr(size);
        std::vector<std::vector<PetscInt>> cols(size);
        std::vector<std::vector<PetscScalar>> vals(size);

        for (int r = 0; r < size; r++)
        {
            PetscInt num_rows = core_end[r] - core_start[r];
            rowptr[r].resize(num_rows + 1, 0);

            // pass 1: build row pointer (counts) by reading the source rows
            for (PetscInt i = 0; i < num_rows; i++)
            {
                PetscInt global_row = core_start[r] + i;
                PetscInt ncols;
                MatGetRow(Source, global_row, &ncols, NULL, NULL);
                rowptr[r][i + 1] = rowptr[r][i] + ncols;
                MatRestoreRow(Source, global_row, &ncols, NULL, NULL);
            }

            PetscInt nnz = rowptr[r][num_rows];
            cols[r].resize(nnz);
            vals[r].resize(nnz);

            // pass 2: fill cols/vals
            for (PetscInt i = 0; i < num_rows; i++)
            {
                PetscInt global_row = core_start[r] + i;
                PetscInt ncols;
                const PetscInt *c;
                const PetscScalar *v;
                MatGetRow(Source, global_row, &ncols, &c, &v);
                for (PetscInt k = 0; k < ncols; k++)
                {
                    cols[r][rowptr[r][i] + k] = c[k];
                    vals[r][rowptr[r][i] + k] = v[k];
                }
                MatRestoreRow(Source, global_row, &ncols, &c, &v);
            }

            if (r == 0)
                continue; // rank 0 inserts its own part directly below, no MPI needed

            // Non-blocking sends: sizes first, then the three payload arrays
            PetscInt sizes[2] = {num_rows, nnz};
            requests.emplace_back();
            MPI_Isend(sizes, 2, MPIU_INT, r, 0, PETSC_COMM_WORLD, &requests.back());

            requests.emplace_back();
            MPI_Isend(rowptr[r].data(), num_rows + 1, MPIU_INT, r, 1, PETSC_COMM_WORLD, &requests.back());

            requests.emplace_back();
            MPI_Isend(cols[r].data(), nnz, MPIU_INT, r, 2, PETSC_COMM_WORLD, &requests.back());

            requests.emplace_back();
            MPI_Isend(vals[r].data(), nnz, MPIU_SCALAR, r, 3, PETSC_COMM_WORLD, &requests.back());
        }

        // ---- rank 0 inserts its own rows directly (no send needed) ----
        {
            PetscInt num_rows = core_end[0] - core_start[0];
            for (PetscInt i = 0; i < num_rows; i++)
            {
                PetscInt global_row = core_start[0] + i;
                PetscInt ncols = rowptr[0][i + 1] - rowptr[0][i];
                MatSetValues(*A, 1, &global_row, ncols,
                             &cols[0][rowptr[0][i]], &vals[0][rowptr[0][i]], INSERT_VALUES);
            }
        }

        // Wait for all sends to complete before the buffers go out of scope
        if (!requests.empty())
            MPI_Waitall((int)requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    }
    else
    {
        // ---- receive this rank's block ----
        PetscInt sizes[2];
        MPI_Recv(sizes, 2, MPIU_INT, 0, 0, PETSC_COMM_WORLD, MPI_STATUS_IGNORE);
        PetscInt num_rows = sizes[0];
        PetscInt nnz = sizes[1];

        std::vector<PetscInt> rowptr(num_rows + 1);
        std::vector<PetscInt> cols(nnz);
        std::vector<PetscScalar> vals(nnz);

        MPI_Recv(rowptr.data(), num_rows + 1, MPIU_INT, 0, 1, PETSC_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(cols.data(), nnz, MPIU_INT, 0, 2, PETSC_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(vals.data(), nnz, MPIU_SCALAR, 0, 3, PETSC_COMM_WORLD, MPI_STATUS_IGNORE);

        for (PetscInt i = 0; i < num_rows; i++)
        {
            PetscInt global_row = Istart + i;
            PetscInt ncols = rowptr[i + 1] - rowptr[i];
            MatSetValues(*A, 1, &global_row, ncols,
                         &cols[rowptr[i]], &vals[rowptr[i]], INSERT_VALUES);
        }
    }

    MatAssemblyBegin(*A, MAT_FINAL_ASSEMBLY); // collective, ALL ranks
    MatAssemblyEnd(*A, MAT_FINAL_ASSEMBLY);
}

/**
 * Distribute a vector that lives only on rank 0 (Source == NULL/unused on
 * every other rank) into a vector *b with the same parallel row layout as
 * Layout (typically x or b from MatCreateVecs on the already-distributed
 * matrix). Collective over PETSC_COMM_WORLD.
 *
 * Unlike DistributeMatrixFromRank0, this does not do its own point-to-point
 * MPI: *b is created with VecDuplicate so its ownership range matches
 * Layout, rank 0 hands PETSc every entry by its global index with
 * VecSetValues, and the collective VecAssemblyBegin/End (called by every
 * rank) is what moves each value to the rank that owns it.
 */
inline void DistributeVectorFromRank0(Vec Layout, Vec Source, Vec *b)
{
    int my_rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &my_rank);

    VecDuplicate(Layout, b); // *b gets Layout's parallel row layout

    if (my_rank == 0)
    {
        PetscInt n;
        VecGetSize(Source, &n);
        const PetscScalar *v;
        VecGetArrayRead(Source, &v);
        std::vector<PetscInt> idx(n);
        for (PetscInt i = 0; i < n; i++)
            idx[i] = i;
        VecSetValues(*b, n, idx.data(), v, INSERT_VALUES);
        VecRestoreArrayRead(Source, &v);
    }
    VecAssemblyBegin(*b); // collective, ALL ranks
    VecAssemblyEnd(*b);
}

#endif /*_RUNAWAY_SOLVER_IFACE_DISTRIBUTE_HPP*/
