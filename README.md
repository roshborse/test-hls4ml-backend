# Vivado AXI-master Backend and Baremetal Application

This is a workspace for testing the integration of the Vivado Accelerator AXI-master backend and baremetal applications. It is a work-in-progress repository, but it should help to converge on a PR for the official hls4ml repository.

The _hls4ml_ fork and branch that we use in this workspace is https://github.com/GiuseppeDiGuglielmo/hls4ml/tree/gdg/axi-m

A few notes
- The backend flow should be controlled from [here](https://github.com/GiuseppeDiGuglielmo/test-hls4ml-backend/blob/main/test_vivado_accelerator.py#L111-L112)
- [This](https://github.com/GiuseppeDiGuglielmo/test-hls4ml-backend/blob/main/test_vivado_accelerator.py#L119-L161) _generator function_ is a draft. It should be general enough to support various models in the future.
- [This call](https://github.com/GiuseppeDiGuglielmo/test-hls4ml-backend/blob/main/test_vivado_accelerator.py#L163) has to be embedded in the backend generation flow (not at the topo level)
- The [sdk](https://github.com/GiuseppeDiGuglielmo/test-hls4ml-backend/tree/main/sdk) directory has to be generated on the fly, right now it is hardcoded at top level.

## Quick Run
```
make run
cd sdk 
make clean sdk gui
```
