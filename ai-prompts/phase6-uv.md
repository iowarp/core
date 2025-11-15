I want this software to be easy to install for people. It should be just one click.

I'm hoping that pip would work here. I want an installer that builds from source
when we do pip install. This is kind of an example: https://github.com/ornladios/ADIOS2/blob/master/pyproject.toml

We use cmake for building. Our main dependencies are mpi, hdf5, zeromq. When building, 
we should disable all tests and benchmarks for now. 

Try making such an installer