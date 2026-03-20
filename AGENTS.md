The following repositories are useful:
- /Users/joao/github/sip: contains the base solver.
- /Users/joao/github/sip_python: contains the python bindings of the solver.
- /Users/joao/github/qpsolvers: contains the integration of sip in the qpsolvers library (generic library for solving QPs with different solvers).
- /Users/joao/github/qpbenchmark: contains the benchmarking code for qpsolvers.
- /Users/joao/github/maros_meszaros_qpbenchmark: contains a specific benchmark used with qpbenchmark.
- /Users/joao/primal-dual-lipa: while not directly relevant, contains some interesting code that may be helpful to port to sip.

We want to improve the performance of the sip solver in qpbenchmark.
Note that some problems may be infeasible; it would be nice to be able to detect them.
Whenever you need to check if e.g. a problem is infeasible or how hard it should be to solve, you can use the PIQP solver, which pretty much does amazingly well at this benchmark.
Note that we are operating inside of the virtual environment /Users/joao/github/qpsolvers/.venv.

The workflow to make changes is roughly this:
1. Change the code in the sip folder.
2. Install the latest sip_python (which already points to the local sip folder as its dependency) to the current venv.
3. Make any required changes to the qpsolvers/qpbenchmark integrations, and also install them.
4. Run the benchmarks, by running the command below from /Users/joao/github/maros_meszaros_qpbenchmark.

Command for running benchmarks:
```
qpbenchmark --results-path /Users/joao/scratch/qpbenchmark_results/dummy_out_sip.csv ./maros_meszaros.py run --solver sip
```

Note that the benchmarks are quite expensive to run, so as much as possible it's nice to iterate on single problems that may be causing issues.
This also avoids the possibility that the OS pauses the process for excessive memory usage.
Generally, especially until SIP is doing better in the benchmarks, we should also try to use fairly tight iteration limits, to prevent the benchmarks taking forever to run. Also, having the benchmarks run fast is perhaps even more important than doing amazingly well, as it allows for faster iteration/improvement cycles.

Ideally, make sure to keep the benchmark logs and per-problem-solve logs in some files that can be easily consulted.
