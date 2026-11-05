/**
 * Memory used by this process (resident set size), and a log of it at
 * named stages of the construction of a simulation (see Process.cpp and
 * Settings/EquationSystem.cpp), which can be read back e.g. by tests.
 */
#ifndef _DREAM_SETTINGS_PROCESS_MEMORY_HPP
#define _DREAM_SETTINGS_PROCESS_MEMORY_HPP

#include <cstdio>
#include <string>
#include <vector>
#include <sys/resource.h>
#include <unistd.h>
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
}

#endif/*_DREAM_SETTINGS_PROCESS_MEMORY_HPP*/
