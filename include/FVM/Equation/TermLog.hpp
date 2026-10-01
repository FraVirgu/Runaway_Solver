#ifndef _DREAM_FVM_EQUATION_TERM_LOG_HPP
#define _DREAM_FVM_EQUATION_TERM_LOG_HPP

/**
 * Helpers for the term-level logs (-dream_log_rebuild and -dream_log_matrix).
 *
 * The two switches are set by DREAM::Solver::RebuildTerms() and
 * DREAM::Solver::BuildMatrix() around their calls into the operators, and
 * read by Operator and AdvectionDiffusionTerm, which print one line for
 * every term they execute.
 */

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cxxabi.h>
#include <string>
#include <typeinfo>

namespace DREAM::FVM::TermLog
{
    // True while Solver::RebuildTerms() should log the terms it rebuilds
    inline bool &rebuild() { static bool v = false; return v; }
    // True while Solver::BuildMatrix() should log the terms it inserts
    inline bool &matrix() { static bool v = false; return v; }

    typedef std::chrono::steady_clock clk;
    inline double ms(clk::time_point a, clk::time_point b)
    { return std::chrono::duration<double, std::milli>(b - a).count(); }

    /** Demangled class name, without the namespace. */
    inline std::string TypeName(const std::type_info &ti)
    {
        int status = 0;
        char *d = abi::__cxa_demangle(ti.name(), nullptr, nullptr, &status);
        std::string s = (status == 0 && d != nullptr) ? d : ti.name();
        std::free(d);
        const size_t pos = s.rfind("::");
        return (pos == std::string::npos) ? s : s.substr(pos + 2);
    }

    /** Short tag for the terms of the kinetic equation, "" for the others. */
    inline std::string Tag(const std::string &cls)
    {
        if (cls == "ElectricFieldTerm")          return "A_E     ";
        if (cls == "SlowingDownTerm")            return "A_C     ";
        if (cls == "SynchrotronTerm")            return "A_s     ";
        if (cls == "EnergyDiffusionTerm")        return "D_pp    ";
        if (cls == "PitchScatterTerm")           return "D_xi,xi ";
        if (cls == "ElectricFieldDiffusionTerm") return "D_E     ";
        return "";
    }

    /** Label of a term: tag (if any) and its name, or its class name if it has none. */
    template <class T>
    std::string Label(const T *term)
    {
        const std::string cls = TypeName(typeid(*term));
        std::string name = term->GetName();
        if (name.empty() || name == "<NOT SET>")
            name = cls;
        return Tag(cls) + name;
    }

    /** Label of an object without a name (for example a boundary condition). */
    template <class T>
    std::string ClassLabel(const T *obj) { return TypeName(typeid(*obj)); }
}

#endif /*_DREAM_FVM_EQUATION_TERM_LOG_HPP*/
