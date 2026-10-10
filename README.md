# Entanglement Entropy Evolution in Shor's Algorithm

This repository contains experimental and research code exploring quantum information concepts, specifically focusing on entanglement entropy calculations and quantum circuit simulations (such as the Inverse Quantum Fourier Transform in `Entropy/src/IQFT.py`) in the context of Shor's algorithm. This was developed in 2020 as part of Simone Amadio Master Thesis in DSSC curriculum with UniTS, SISSA and ICTP, under the supervision of Professor Antonello Scardicchio.

---

## Thesis Reference & Academic Context
This repository forms part of the research and simulations developed for the Master's Thesis:

* **Title:** *Entanglement Entropy evolution in Shor's algorithm*
* **Author:** Simone Amadio[cite: 1]
* **Advisor:** Prof. Antonello Scardicchio[cite: 1]
* **Affiliations:** 
  * **ICTP** – The Abdus Salam International Centre for Theoretical Physics[cite: 1]
  * **SISSA** – Scuola Internazionale Superiore di Studi Avanzati
  * **Università degli Studi di Trieste** (UniTS)
* **Document Link:** [Read Thesis PDF (Local Copy)](docs/tesi_entropy.pdf)

---

## Abstract
> In recent years, the advances of quantum computing and the increased availability of quantum hardware allowed us to discover several instances of quantum speedup and quantum advantage[cite: 12]. Historically, Shor's algorithm was one of the first quantum algorithms to suggest a speedup with respect to the classical case[cite: 12]. 
> 
> Despite the presence of more and more experimental instances, the deep reasons behind the quantum speedup are still difficult to grasp and define[cite: 12]. Quantum computing is inherently parallel and it differs from classical computation in two main aspects: state superposition and existence of entanglement[cite: 12]. This work aims to compute and analyze the entanglement in Shor's algorithm, specifically through the means of entanglement entropy[cite: 12]. After discussing all the preliminary concepts and the known results, our work is focused on simulating Shor's algorithm for up to 27 qubits in order to estimate the distribution of entanglement entropy across all balanced bipartitions of the system[cite: 12].
> 
> Our results show that the entanglement entropy behaviour is strongly affected by the algorithm inputs, specifically we discover a strong role of the order $r$: distributions for generic balanced bipartitions follow a trend similar to the registers bipartition when $r$ is a power of two[cite: 12]. When this is not the case, the entanglement distribution gets on average more and more multipartite with $r$ and the behaviour of the distributions is affected even by the input number $N$ which we want to factorize[cite: 12, 13].

---

## Repository Structure & Key Implementations
- **`Entropy/src/IQFT.py`**: Implements choice selection between a tensor-matrix Inverse Quantum Fourier Transform (IQFT) and a Qiskit statevector implementation, constructs Fourier-matrix rows, and handles qubit-order inversion.

## Citation
If you use this codebase or refer to the theoretical framework in your research, please cite the Master's Thesis as follows:

```bibtex
@mastersthesis{Amadio2020ShorEntanglement,
  author       = {Amadio, Simone},
  title        = {Entanglement Entropy evolution in Shor's algorithm},
  school       = {University of Trieste, SISSA, and ICTP},
  year         = {2020},
  type         = {Master's Thesis},
  note         = {Advisor: Prof. Antonello Scardicchio. Available at: https://github.com/samadio/entanglement_entropy}
}

```

## Licensing

* **Source Code:** All code in this repository is open-source software made available under the MIT License (Copyright (c) 2020 Simone Amadio).
* **Thesis Document (`docs/tesi_entropy.pdf`):** The Master's Thesis document is © 2020 Simone Amadio. All rights reserved.
