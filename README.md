# RNN_IFWI
## This is the demo code for the MLP-based implicit full waveform inversion
#### The elastic models used in the notbook can be downloaded from the following link.
#### https://drive.google.com/drive/folders/1HZZ7ofv0_KDJMuf2aQzNhaTBl4x6MAwM?usp=sharing

We introduce and analyze the elastic implicit full waveform inversion (EIFWI) of seismic data, which uses neural networks to generate elastic models and perform full waveform inversion. EIFWI carries out inversion by linking two main networks: a neural network that generates elastic models and a recurrent neural network to perform the modeling. The approach is distinct from conventional waveform inversion in two key ways. First, it reduces reliance on accurate initial models relative to conventional FWI. Instead, it invokes general information about the target area, for instance, estimates of means and standard deviations of medium properties in the target area or, alternatively, well-log information in the target area. Second, iterative updating directly affects the weights in the neural network rather than the elastic model. Elastic models can be generated in the first part of the EIFWI process through the use of a multilayer perceptron (MLP) network. Numerical testing is suggestive that the MLP-based EIFWI approach in principle builds accurate models in the absence of an explicit initial model.

## Citing
If you would like to cite this code, please use our papers:
```
@article{sun2023implicit,
  title={Implicit seismic full waveform inversion with deep neural representation},
  author={Sun, Jian and Innanen, Kristopher and Zhang, Tianze and Trad, Daniel},
  journal={Journal of Geophysical Research: Solid Earth},
  volume={128},
  number={3},
  pages={e2022JB025964},
  year={2023},
  publisher={Wiley Online Library}
}

@article{zhang2023multilayer,
  title={Multilayer perceptron and Bayesian neural network based elastic implicit full waveform inversion},
  author={Zhang, Tianze and Sun, Jian and Trad, Daniel and Innanen, Kristopher},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2023},
  publisher={IEEE}
}
```

