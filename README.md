## Crosstalk Aware HyperQ
This repository is a modified version of the Artifact for OSDI'25 "Quantum Virtual Machines" from Runzhou Tao, Hongzheng Zhu, Jason Nieh, Jianan Yao and Ronghui Gu. 

### Changes
We changed the space scheduling accounting for crosstalk noise compatibility. 
- Optimized initial mapping
- Implementation of a derived version from the iterative mapping optimization algorithm in [TRIM: crossTalk-awaRe qubIt Mapping for multiprogrammed quantum systems](https://ieeexplore.ieee.org/document/10234256).

### Experimental Results
Our three experimental runs are included in the `Results/` directory. 

### Usage
For general usage instructions, refer to [https://github.com/1640675651/HyperQ](https://github.com/1640675651/HyperQ). 

Additions: 
- Set `buffer_qubits=False` in elastic_qvm creation when omitting qubit separations between circuits through the input coupling map.
- Define crosstalk-sensitive qubit pairs for a specific input coupling map and pass into HypervisorBackend instance creation. 
  
### License
The code in `ModifiedHyperQ/` remains under the original license.
