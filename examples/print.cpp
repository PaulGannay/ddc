#include <iostream>
#include <string>
#include <iomanip>

#include <ddc/ddc.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

// Name of the axis
struct DDimX
{
};
struct DDimY
{
};
struct DDimZ
{
};

using cell = double;

void print3d(){
    unsigned const length = 100;
    unsigned const height = 100;
    unsigned const depth = 100;

    ddc::DiscreteDomain<DDimX> const domain_x
            = ddc::init_trivial_bounded_space(ddc::DiscreteVector<DDimX>(length));
    ddc::DiscreteDomain<DDimY> const domain_y
            = ddc::init_trivial_bounded_space(ddc::DiscreteVector<DDimY>(height));
    ddc::DiscreteDomain<DDimZ> const domain_z
            = ddc::init_trivial_bounded_space(ddc::DiscreteVector<DDimZ>(depth));

    ddc::DiscreteDomain<DDimX, DDimY, DDimZ> const domain_xyz(domain_x, domain_y, domain_z);

    ddc::Chunk cells_in_dev_alloc("cells_in_dev", domain_xyz, ddc::DeviceAllocator<cell>());
    ddc::ChunkSpan const cells_in = cells_in_dev_alloc.span_view();

    Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);
    ddc::parallel_for_each(
            domain_xyz,
            KOKKOS_LAMBDA(ddc::DiscreteElement<DDimX, DDimY, DDimZ> const ixyz) {
                auto generator = random_pool.get_state();
                cells_in(ixyz) = generator.drand(-1.,1.);
                //cells_in(ixyz) = 1;
                random_pool.free_state(generator);
            });

    print (std::cout, cells_in);
    print_chunk_info(std::cout, cells_in);
}

void print0d() {
    ddc::DiscreteDomain<> const domain_full;

    ddc::Chunk cells_in_dev_alloc("cells_in_dev", domain_full, ddc::DeviceAllocator<cell>());
    ddc::ChunkSpan const cells_in = cells_in_dev_alloc.span_view();

    Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);
    ddc::parallel_for_each(
            domain_full,
            KOKKOS_LAMBDA(ddc::DiscreteElement<> const i) {
                auto generator = random_pool.get_state();
                cells_in(i) = generator.drand(-1.,1.);
                random_pool.free_state(generator);
            });

    std::cout << cells_in << std::endl;
}

void print2d() {
    unsigned const dim0 = 3;
    unsigned const dim1 = 3;

    struct DDim0 {};
    struct DDim1 {};

    ddc::DiscreteDomain<DDim0> const domain_0
            = ddc::init_trivial_bounded_space(ddc::DiscreteVector<DDim0>(dim0));
    ddc::DiscreteDomain<DDim1> const domain_1
            = ddc::init_trivial_bounded_space(ddc::DiscreteVector<DDim1>(dim1));

    ddc::DiscreteDomain<DDim0, DDim1> 
		const domain_full(domain_0, domain_1);

    using cell = int;

    ddc::Chunk cells_in_dev_alloc("cells_in_dev", domain_full, ddc::DeviceAllocator<cell>());
    ddc::ChunkSpan const cells_in = cells_in_dev_alloc.span_view();

    Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);
    ddc::parallel_for_each(
            domain_full,
            KOKKOS_LAMBDA(ddc::DiscreteElement<DDim0, DDim1> const i) {
                auto generator = random_pool.get_state();
                cells_in(i) = generator.drand(-10000.,10000.);
                //cells_in(i) = 1;
                random_pool.free_state(generator);
            });

    std::cout << cells_in << std::endl;
    std::cout << std::hex << cells_in << std::endl;
}

void print6d(){
    unsigned const dim0 = 1;
    unsigned const dim1 = 2;
    unsigned const dim2 = 2;
    unsigned const dim3 = 2;
    unsigned const dim4 = 2;
    unsigned const dim5 = 1;

	struct DDim0 {};
	struct DDim1 {};
	struct DDim2 {};
	struct DDim3 {};
	struct DDim4 {};
	struct DDim5 {};

    ddc::DiscreteDomain<DDim0> const domain_0
            = ddc::init_trivial_bounded_space(ddc::DiscreteVector<DDim0>(dim0));
    ddc::DiscreteDomain<DDim1> const domain_1
            = ddc::init_trivial_bounded_space(ddc::DiscreteVector<DDim1>(dim1));
    ddc::DiscreteDomain<DDim2> const domain_2
            = ddc::init_trivial_bounded_space(ddc::DiscreteVector<DDim2>(dim2));
    ddc::DiscreteDomain<DDim3> const domain_3
            = ddc::init_trivial_bounded_space(ddc::DiscreteVector<DDim3>(dim3));
    ddc::DiscreteDomain<DDim4> const domain_4
            = ddc::init_trivial_bounded_space(ddc::DiscreteVector<DDim4>(dim4));
    ddc::DiscreteDomain<DDim5> const domain_5
            = ddc::init_trivial_bounded_space(ddc::DiscreteVector<DDim5>(dim5));

    ddc::DiscreteDomain<DDim0, DDim1, DDim2, DDim3, DDim4, DDim5> 
		const domain_full(domain_0, domain_1, domain_2, domain_3, domain_4, domain_5);

    ddc::Chunk cells_in_dev_alloc("cells_in_dev", domain_full, ddc::DeviceAllocator<cell>());
    ddc::ChunkSpan const cells_in = cells_in_dev_alloc.span_view();

    Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);
    ddc::parallel_for_each(
            domain_full,
            KOKKOS_LAMBDA(ddc::DiscreteElement<DDim0, DDim1, DDim2, DDim3, DDim4, DDim5> const i) {
                auto generator = random_pool.get_state();
                cells_in(i) = generator.drand(-1.,1.);
                //cells_in(i) = 1;
                random_pool.free_state(generator);
            });

    print (std::cout, cells_in);

    //print_chunk_info(std::cout, cells_in);
}

void print4d(){
    unsigned const dim0 = 2;
    unsigned const dim1 = 2;
    unsigned const dim2 = 2;
    unsigned const dim3 = 2;

	struct DDim0 {};
	struct DDim1 {};
	struct DDim2 {};
	struct DDim3 {};

    ddc::DiscreteDomain<DDim0> const domain_0
            = ddc::init_trivial_bounded_space(ddc::DiscreteVector<DDim0>(dim0));
    ddc::DiscreteDomain<DDim1> const domain_1
            = ddc::init_trivial_bounded_space(ddc::DiscreteVector<DDim1>(dim1));
    ddc::DiscreteDomain<DDim2> const domain_2
            = ddc::init_trivial_bounded_space(ddc::DiscreteVector<DDim2>(dim2));
    ddc::DiscreteDomain<DDim3> const domain_3
            = ddc::init_trivial_bounded_space(ddc::DiscreteVector<DDim3>(dim3));

    ddc::DiscreteDomain<DDim0, DDim1, DDim2, DDim3> 
		const domain_full(domain_0, domain_1, domain_2, domain_3);

    ddc::Chunk cells_in_dev_alloc("cells_in_dev", domain_full, ddc::DeviceAllocator<cell>());
    ddc::ChunkSpan const cells_in = cells_in_dev_alloc.span_view();

    Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);
    ddc::parallel_for_each(
            domain_full,
            KOKKOS_LAMBDA(ddc::DiscreteElement<DDim0, DDim1, DDim2, DDim3> const i) {
                auto generator = random_pool.get_state();
                cells_in(i) = generator.drand(-1.,1.);
                //cells_in(i) = 1;
                random_pool.free_state(generator);
            });

    print (std::cout, cells_in);

    //print_chunk_info(std::cout, cells_in);
}

int main () 
{
  Kokkos::ScopeGuard const kokkos_scope;
  ddc::ScopeGuard const ddc_scope;

	print4d();
}
