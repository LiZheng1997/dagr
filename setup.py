from distutils.core import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

nvcc_flags = [
    "-O3",
    "--expt-relaxed-constexpr",
    "-allow-unsupported-compiler",
]

setup(
    name='dagr',
    packages=['dagr'],
    package_dir={'':'src'},
    ext_modules=[
        CUDAExtension(name='asy_tools',
                      sources=['src/dagr/asynchronous/asy_tools/main.cu'],
                      extra_compile_args={"cxx": [], "nvcc": nvcc_flags}),
        CUDAExtension(name="ev_graph_cuda",
                      sources=['src/dagr/graph/ev_graph.cu'],
                      extra_compile_args={"cxx": [], "nvcc": nvcc_flags})
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
