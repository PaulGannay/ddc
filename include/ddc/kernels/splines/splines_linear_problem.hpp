// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cassert>
#include <cstddef>
#include <iomanip>
#include <ostream>

#include <Kokkos_Core.hpp>

namespace ddc::detail {

/**
 * @brief The parent class for linear problems dedicated to the computation of spline approximations.
 *
 * Store a square matrix and provide method to solve a multiple right-hand sides linear problem.
 * Implementations may have different storage formats, filling methods and multiple right-hand sides linear solvers.
 */
template <class ExecSpace>
class SplinesLinearProblem
{
public:
    /// @brief The type of a Kokkos::View storing multiple right-hand sides.
    using MultiRHS = Kokkos::View<double**, Kokkos::LayoutRight, ExecSpace>;

private:
    std::size_t m_size;

protected:
    explicit SplinesLinearProblem(const std::size_t size) : m_size(size) {}

public:
    SplinesLinearProblem(SplinesLinearProblem const& x) = delete;

    SplinesLinearProblem(SplinesLinearProblem&& x) = delete;

    /// @brief Destruct
    virtual ~SplinesLinearProblem() = default;

    SplinesLinearProblem& operator=(SplinesLinearProblem const& x) = delete;

    SplinesLinearProblem& operator=(SplinesLinearProblem&& x) = delete;

    /**
     * @brief Get an element of the matrix at indexes i, j. It must not be called after `setup_solver`.
     *
     * @param i The row index of the desired element.
     * @param j The column index of the desired element.
     *
     * @return The value of the element of the matrix.
     */
    virtual double get_element(std::size_t i, std::size_t j) const = 0;

    /**
     * @brief Set an element of the matrix at indexes i, j. It must not be called after `setup_solver`.
     *
     * @param i The row index of the set element.
     * @param j The column index of the set element.
     * @param aij The value to set in the element of the matrix.
     */
    virtual void set_element(std::size_t i, std::size_t j, double aij) = 0;

    /**
     * @brief Perform a pre-process operation on the solver. Must be called after filling the matrix.
     */
    virtual void setup_solver() = 0;

    /**
     * @brief Solve the multiple right-hand sides linear problem Ax=b or its transposed version A^tx=b inplace.
     *
     * @param[in, out] b A 2D Kokkos::View storing the multiple right-hand sides of the problem and receiving the corresponding solution.
     * @param transpose Choose between the direct or transposed version of the linear problem.
     */
    virtual void solve(MultiRHS b, bool transpose) const = 0;

    /**
     * @brief Get the size of the square matrix in one of its dimensions.
     *
     * @return The size of the matrix in one of its dimensions.
     */
    std::size_t size() const
    {
        return m_size;
    }

    /**
     * @brief Get the required number of rows of the multi-rhs view passed to solve().
     *
     * Implementations may require a number of rows larger than what `size` returns for optimization purposes.
     *
     * @return The required number of rows of the multi-rhs view. It is guaranteed to be greater or equal to `size`.
     */
    std::size_t required_number_of_rhs_rows() const
    {
        std::size_t const nrows = impl_required_number_of_rhs_rows();
        assert(nrows >= size());
        return nrows;
    }

private:
    virtual std::size_t impl_required_number_of_rhs_rows() const
    {
        return m_size;
    }
};

/**
 * @brief Prints the matrix of a SplinesLinearProblem in a std::ostream. It must not be called after `setup_solver`.
 *
 * @param[out] os The stream in which the matrix is printed.
 * @param[in] linear_problem The SplinesLinearProblem of the matrix to print.
 *
 * @return The stream in which the matrix is printed.
**/
template <class ExecSpace>
std::ostream& operator<<(std::ostream& os, SplinesLinearProblem<ExecSpace> const& linear_problem)
{
    std::size_t const n = linear_problem.size();
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            os << std::fixed << std::setprecision(3) << std::setw(10)
               << linear_problem.get_element(i, j);
        }
        os << "\n";
    }
    return os;
}

} // namespace ddc::detail
