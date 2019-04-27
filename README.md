# Neuronal Simulation and Evolution
The project aims to simulate neuronal ensembles and perform evolutionary algorithms to optimize the performance of the ensemble with respect to a fitness function.


## Simulation
Results of simple hebbian potentiation of synaptic weights using Triplet spike-time-dependent-plasticity (STDP) rules and [AdEx neurons](http://www.scholarpedia.org/article/Adaptive_exponential_integrate-and-fire_model).

Reproducd by [simple_association.py][simple_asso_code]:
```python
python simple_association.py
```

Additional synaptic rules such as heterosynaptic consolidation and homeostatic regulation ([Zenke et al. 2014](https://www.nature.com/articles/ncomms7922)) are implemented but disabled in this example. They can be found at [neuroevolutioner/SynapticDynamics.py][Syn_dynamics_code]

![simple_asso_fig]

```
Descriptions of plots from top to bottom:

Plot# 1: Soma potentials of neuron 1 and 2, with firing moments marked.
Plot# 2: Increase of weights during pairing. (weight_1 = weight from neuron 1 to 2)
Plot# 3: Synaptic current. (syn_current_1 = input current into neuron 1) 
Plot# 4: External current applied on neuron 1 and 2.
Plot# 5-7: Traces of LTP, LTP_slow, LTD, for neuron 1 and 2,
Plot# 8: The variable of short-term adaptation dynamics in AdEx Neuron.  
```

After co-firing activities (Plot#1, x-axis between 3 - 4), weight between two neurons potentiated.

Note that the weight increase under STDP is unbounded. Regulation by homeostatic/heterosynaptic dynamics is required for stability.

## Evolution (In progress)

Preview of implementations: [Genetics.py][genetics_module] and [Evolution.py][evolution_module].

Example run code: 
```python
python DA_analysis.py
```  

## References

1. Brette and Gerstner 2005, Adaptive exponential integrate-and-fire model as an effective description of neuronal activity (https://www.ncbi.nlm.nih.gov/pubmed/16014787)
2. Pfister and Gerstner 2006, Triplets of spikes in a model of spike timing-dependent plasticity
(https://www.ncbi.nlm.nih.gov/pubmed/16988038)
3. Zenke, Agnes and Gerstner 2015, Diverse synaptic plasticity mechanisms orchestrated to form and retrieve memories in spiking neural networks (https://www.nature.com/articles/ncomms7922)


[simple_asso_fig]: figs/simple_association.png
[simple_asso_code]: simple_association.py
[evolution_code]: DA_analysis.py
[evolution_module]: neuroevolutioner/Evolution.py
[genetics_module]: neuroevolutioner/Genetics.py
[Syn_dynamics_code]: neuroevolutioner/SynapticDynamics.py
