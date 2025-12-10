### Crosstalk Aware HyperQ
This Repository is a modified version of the Artifact for OSDI'25 "Quantum Virtual Machines" from Runzhou Tao, Hongzheng Zhu, Jason Nieh, Jianan Yao and Ronghui Gu. The space scheduling algorithm was changed accounting for crosstalk noise compatibility of compiled circuits. 

### Usage
For general usage instructions, refer to [https://github.com/1640675651/HyperQ](https://github.com/1640675651/HyperQ). 

Set buffer_qubits to False in elastic_qvm creation when omitting qubit separations between circuits in ModifiedHyperQ/benchmark.py. 

### License
The code in `ModifiedHyperQ/` remains under the original license.
