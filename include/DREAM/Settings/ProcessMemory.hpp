/**
 * Memory used by this process (resident set size), and a log of it at
 * named stages of the construction of a simulation (see Process.cpp,
 * Settings/EquationSystem.cpp and the first time step of
 * SolverLinearlyImplicit::Solve()), which can be read back e.g. by tests
 * and is printed by dreami with -dream_log_memory.
 */
#ifndef _DREAM_SETTINGS_PROCESS_MEMORY_HPP
#define _DREAM_SETTINGS_PROCESS_MEMORY_HPP

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>
#include <sys/resource.h>
#include <unistd.h>
#include <petscsys.h>
#include "FVM/config.h"

namespace DREAM {
    /**
     * Resident set size of this process [bytes]: current, from
     * /proc/self/statm (second field, in pages), and peak, from getrusage
     * (ru_maxrss, in kB). (PetscMemoryGetCurrentUsage() is not used:
     * without PETSC_USE_PROCFS_FOR_SIZE it returns ru_maxrss, the peak.)
     */
    inline void GetProcessMemory(real_t &current, real_t &peak) {
        current = 0;
        if (FILE *statm = fopen("/proc/self/statm", "r")) {
            unsigned long size, resident;
            if (fscanf(statm, "%lu %lu", &size, &resident) == 2)
                current = (real_t)resident*sysconf(_SC_PAGESIZE);
            fclose(statm);
        }
        struct rusage usage;
        getrusage(RUSAGE_SELF, &usage);
        peak = (real_t)usage.ru_maxrss*1024.0;
    }

    struct ProcessMemoryRecord { std::string stage; real_t current, peak; };

    // Memory at the stages recorded so far, in order
    inline std::vector<ProcessMemoryRecord> &ProcessMemoryLog() {
        static std::vector<ProcessMemoryRecord> log;
        return log;
    }

    // Record the memory used now, at the end of the given stage
    inline void RecordProcessMemory(const std::string &stage) {
        real_t current, peak;
        GetProcessMemory(current, peak);
        ProcessMemoryLog().push_back({stage, current, peak});
    }

    /**
     * Print the recorded stages as a table, one row per stage, with the
     * resident set size over the ranks of PETSC_COMM_WORLD (min, max, sum),
     * the largest increase on a rank since the previous stage, and the
     * largest peak. Collective: every rank must call it; rank 0 writes the
     * table to each non-null file in 'files'. Only the stages recorded on
     * every rank are printed (the names are those of rank 0).
     *
     * The format is read by summarize_memory.py: keep the header line
     * "Resident set size at the end of each stage" and the column order.
     */
    inline void PrintProcessMemoryLog(const std::vector<FILE*> &files) {
        int rank, size;
        MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
        MPI_Comm_size(PETSC_COMM_WORLD, &size);

        const std::vector<ProcessMemoryRecord> &stages = ProcessMemoryLog();
        int nStages = (int)stages.size(), nStagesMin;
        MPI_Allreduce(&nStages, &nStagesMin, 1, MPI_INT, MPI_MIN, PETSC_COMM_WORLD);

        // [rank][stage][current, peak]
        std::vector<real_t> mine(2*nStagesMin);
        for (int k = 0; k < nStagesMin; k++) {
            mine[2*k]   = stages[k].current;
            mine[2*k+1] = stages[k].peak;
        }
        std::vector<real_t> all(rank == 0 ? 2*nStagesMin*size : 0);
        MPI_Gather(mine.data(), 2*nStagesMin, MPIU_REAL, all.data(), 2*nStagesMin, MPIU_REAL, 0, PETSC_COMM_WORLD);

        if (rank != 0)
            return;

        const real_t MB = 1024.0*1024.0;
        for (FILE *f : files) {
            if (f == nullptr)
                continue;
            fprintf(f, "\nResident set size at the end of each stage of the construction [MB] (%d rank%s)\n",
                size, size == 1 ? "" : "s");
            fprintf(f, "  min/max/sum: over the ranks; added: largest increase on a rank since the previous stage;"
                       " peak max: largest peak RSS of a rank so far\n");
            fprintf(f, "%-48s  %10s  %10s  %12s  %10s  %10s\n",
                "stage", "RSS min", "RSS max", "RSS sum", "added", "peak max");
            for (int k = 0; k < nStagesMin; k++) {
                real_t lo = 0, hi = 0, sum = 0, added = 0, pk = 0;
                for (int r = 0; r < size; r++) {
                    const real_t c = all[2*nStagesMin*r + 2*k];
                    const real_t prev = (k > 0 ? all[2*nStagesMin*r + 2*(k-1)] : 0);
                    lo = (r == 0 ? c : std::min(lo, c));
                    hi = std::max(hi, c);
                    sum += c;
                    added = std::max(added, c - prev);
                    pk = std::max(pk, all[2*nStagesMin*r + 2*k+1]);
                }
                fprintf(f, "%-48s  %10.1f  %10.1f  %12.1f  %10.1f  %10.1f\n",
                    stages[k].stage.c_str(), lo/MB, hi/MB, sum/MB, added/MB, pk/MB);
            }
            fprintf(f, "End of resident set size table\n");
            fflush(f);
        }
    }
}

#endif/*_DREAM_SETTINGS_PROCESS_MEMORY_HPP*/
