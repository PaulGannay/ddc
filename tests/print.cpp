// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

inline namespace anonymous_namespace_workaround_print_cpp {

struct DDim0
{
};
struct DDim1
{
};
struct DDim2
{
};
struct DDim3
{
};
struct DDim4
{
};
struct DDim5
{
};

} // namespace anonymous_namespace_workaround_print_cpp

TEST(Print, Simple2DChunk)
{
    using cell = double;

    unsigned const dim0 = 2;
    unsigned const dim1 = 2;

    ddc::DiscreteDomain<DDim0> const domain_0
            = ddc::init_trivial_bounded_space(ddc::DiscreteVector<DDim0>(dim0));
    ddc::DiscreteDomain<DDim1> const domain_1
            = ddc::init_trivial_bounded_space(ddc::DiscreteVector<DDim1>(dim1));

    ddc::DiscreteDomain<DDim0, DDim1> const domain_2d(domain_0, domain_1);

    ddc::Chunk cells_in_dev_alloc("cells_in_dev", domain_2d, ddc::DeviceAllocator<cell>());
    ddc::ChunkSpan const cells_in = cells_in_dev_alloc.span_view();

    Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);
    ddc::parallel_for_each(
            domain_2d,
            KOKKOS_LAMBDA(ddc::DiscreteElement<DDim0, DDim1> const i) {
                auto generator = random_pool.get_state();
                cells_in(i) = generator.drand(-1., 1.);
                random_pool.free_state(generator);
            });

    {
        std::stringstream ss;
        ss << cells_in;
        EXPECT_EQ(
                "[[ 0.470186 -0.837013]\n"
                " [ 0.439832  0.347536]]",
                ss.str());
    }
    {
        std::stringstream ss;
        ss << std::setprecision(2) << cells_in;
        EXPECT_EQ(
                "[[ 0.47 -0.84]\n"
                " [ 0.44  0.35]]",
                ss.str());
    }
    {
        std::stringstream ss;
        ss << std::hexfloat << cells_in;
        EXPECT_EQ(
                "[[ 0x1.e178596d8678cp-2 -0x1.ac8cf5563aa03p-1]\n"
                " [ 0x1.c263537950d98p-2   0x1.63e08ca1dfd3p-2]]",
                ss.str());
    }
    {
        std::stringstream ss;
        ss << std::scientific << cells_in;
        EXPECT_EQ(
                "[[ 4.701857e-01 -8.370129e-01]\n"
                " [ 4.398320e-01  3.475363e-01]]",
                ss.str());
    }
}



TEST(Print, 3DChunkSpan)
{
    using cell = double;

    unsigned const dim0 = 3;
    unsigned const dim1 = 3;
    unsigned const dim2 = 3;

    ddc::DiscreteDomain<DDim0> const domain_0
            = ddc::init_trivial_bounded_space(ddc::DiscreteVector<DDim0>(dim0));
    ddc::DiscreteDomain<DDim1> const domain_1
            = ddc::init_trivial_bounded_space(ddc::DiscreteVector<DDim1>(dim1));
    ddc::DiscreteDomain<DDim2> const domain_2
            = ddc::init_trivial_bounded_space(ddc::DiscreteVector<DDim2>(dim2));

    ddc::DiscreteDomain<DDim0, DDim1, DDim2> const domain_3d(domain_0, domain_1, domain_2);

    ddc::Chunk cells_in_dev_alloc("cells_in_dev", domain_3d, ddc::DeviceAllocator<cell>());
    ddc::ChunkSpan const cells_in = cells_in_dev_alloc.span_view();

    Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);
    ddc::parallel_for_each(
            domain_3d,
            KOKKOS_LAMBDA(ddc::DiscreteElement<DDim0, DDim1, DDim2> const i) {
                auto generator = random_pool.get_state();
                cells_in(i) = generator.drand(-1., 1.);
                random_pool.free_state(generator);
            });

    {
        std::stringstream ss;
        ss << cells_in;
        EXPECT_EQ(
                "[[[  0.470186  -0.837013   0.439832]\n"
                "  [  0.347536   0.311911  -0.601701]\n"
                "  [ -0.647321   0.465216  -0.304076]]\n"
                "\n"
                " [[ -0.734212  -0.948613  -0.520679]\n"
                "  [  0.133496  -0.760342 -0.0072482]\n"
                "  [ -0.514225   0.723847   0.919229]]\n"
                "\n"
                " [[  0.854303   0.971342   0.958612]\n"
                "  [ -0.855815  -0.510559   0.294396]\n"
                "  [ 0.0678173   0.227127  -0.727128]]]",
                ss.str());
    }
}

TEST(Print, CheckOutput0d)
{
    using cell = double;

    ddc::DiscreteDomain<> const domain_full;

    ddc::Chunk cells_in_dev_alloc("cells_in_dev", domain_full, ddc::DeviceAllocator<cell>());
    ddc::ChunkSpan const cells_in = cells_in_dev_alloc.span_view();

    Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);
    ddc::parallel_for_each(
            domain_full,
            KOKKOS_LAMBDA(ddc::DiscreteElement<> const i) {
                auto generator = random_pool.get_state();
                cells_in(i) = generator.drand(-1., 1.);
                random_pool.free_state(generator);
            });

    {
        std::stringstream ss;
        ss << cells_in;
        EXPECT_EQ("0.470186", ss.str());
    }
}

TEST(Print, 2DChunkElision)
{
    using cell = double;

    unsigned const dim0 = 100;
    unsigned const dim1 = 100;

    ddc::DiscreteDomain<DDim0> const domain_0
            = ddc::init_trivial_bounded_space(ddc::DiscreteVector<DDim0>(dim0));
    ddc::DiscreteDomain<DDim1> const domain_1
            = ddc::init_trivial_bounded_space(ddc::DiscreteVector<DDim1>(dim1));

    ddc::DiscreteDomain<DDim0, DDim1> const domain_2d(domain_0, domain_1);

    ddc::Chunk cells_in_dev_alloc("cells_in_dev", domain_2d, ddc::DeviceAllocator<cell>());
    ddc::ChunkSpan const cells_in = cells_in_dev_alloc.span_view();

    Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);
    ddc::parallel_for_each(
            domain_2d,
            KOKKOS_LAMBDA(ddc::DiscreteElement<DDim0, DDim1> const i) {
                auto generator = random_pool.get_state();
                cells_in(i) = generator.drand(-1., 1.);
                random_pool.free_state(generator);
            });

    {
        std::stringstream ss;
        ss << cells_in;
        EXPECT_EQ(
                "[[  0.470186  -0.837013   0.439832 ...  -0.591551    0.80371   0.645479]\n"
                " [   0.95908   -0.48044 -0.0894193 ...  0.0740702  -0.384853  -0.632442]\n"
                " [ -0.686111  -0.311298   0.987831 ...  -0.440372  -0.926175   0.485239]\n"
                " ...\n"
                " [ -0.672112  -0.895914  -0.794096 ...  -0.765889  0.0821389  -0.348865]\n"
                " [  0.324101   0.340524  -0.147613 ...   0.437029    -0.4914  -0.609299]\n"
                " [  0.476478  -0.361951  -0.610811 ...  -0.283171   0.597408  -0.628835]]",
                ss.str());
    }
}
