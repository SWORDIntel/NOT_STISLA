# KEYSTONE Documentation

The root [README](../README.md) is intentionally written as a high-level introduction. Use this directory for implementation, integration, benchmark, and engineering detail.

## Start here

| Document | Use it for |
|---|---|
| [TECHNICAL_OVERVIEW.md](TECHNICAL_OVERVIEW.md) | Architecture, search model, backend selection, feature matrix, memory behavior, ingestion pipeline, and QIHSE bridge. |
| [STATUS_SUMMARY.md](STATUS_SUMMARY.md) | Current implementation boundary and engineering backlog. |
| [BUILD_MODES.md](BUILD_MODES.md) | Native, scalar, optional Fortran, archive, and feature-controlled builds. |
| [INTEGRATION.md](INTEGRATION.md) | Embedding KEYSTONE into another application or data pipeline. |
| [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md) | Current measurement rules and benchmark results. |
| [ACCELERATOR_CONTRACT.md](ACCELERATOR_CONTRACT.md) | Requirements a GPU/NPU/other accelerator backend must satisfy before it is treated as supported. |
| [TELEMETRY_PROCESSOR.md](TELEMETRY_PROCESSOR.md) | Telemetry processor implementation and usage. |

## Benchmark records

The repository also retains historical and phase-specific benchmark reports:

- [benchmark_report.md](benchmark_report.md)
- [benchmark_report_phase5.md](benchmark_report_phase5.md)

Treat host-specific benchmark reports as measurements of the stated machine/build/workload, not as universal performance guarantees.

## Documentation rule

The README should answer:

1. What does KEYSTONE do?
2. Why would an organization use it?
3. Where does it fit into existing infrastructure?
4. What has actually been measured or implemented?
5. Where can a technical reader go deeper?

Detailed backend mechanics, build switches, benchmark methodology, feature matrices, and experimental implementation notes belong under `docs/` so the public entry point stays readable to both technical and non-technical reviewers.
