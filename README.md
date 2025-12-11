## Crosstalk Aware HyperQ
This repository is a modified version of the Artifact for OSDI'25 "Quantum Virtual Machines" from Runzhou Tao, Hongzheng Zhu, Jason Nieh, Jianan Yao, and Ronghui Gu. 

### Changes
We modified the space scheduling to account for program crosstalk noise compatibilities, improving fidelity with:
- Optimized initial mapping
- Implementation of a derived version of the iterative mapping optimization algorithm in [TRIM: crossTalk-awaRe qubIt Mapping for multiprogrammed quantum systems](https://ieeexplore.ieee.org/document/10234256).

### Experimental Results
Our three experimental runs are included in the `Results/` directory. 

### Usage
For general usage instructions, refer to [https://github.com/1640675651/HyperQ](https://github.com/1640675651/HyperQ). 

Changes in usage: 
- Set `buffer_qubits=False` in elastic_qvm creation when omitting qubit separations between circuits in the input coupling map.
- Define crosstalk-sensitive horizontal and vertical qubit pairs for input coupling maps and pass into HypervisorBackend `ct_h` and `ct_v` attributes. 
  
### License
The code in `ModifiedHyperQ/` remains under the original license.

Note: Removing buffer qubits as implemented in `ModifiedHyperQ/benchmark.py` might not work properly, as it did not increase the utilization in our results.
