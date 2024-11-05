# Quantifying Individual Fairness in Continual Federated Learning

This package provides tools and strategies for enhancing individual fairness in continual learning within federated learning settings. It introduces novel fairness metrics and client selection strategies specifically designed to maintain temporal fairness across clients in a continual federated learning (CFL) context.

#Key Features
Fairness Metrics:

Delta Accuracy Fairness (DAF) is tailored to assess and maintain fairness over time in CFL by measuring accuracy and retention disparities among clients.
Client Selection Strategies:

Strategies such as Low Participation, Low Accuracy, and Low Average are implemented to prioritize fairness by addressing knowledge retention inequalities, with comprehensive analysis showing their effectiveness in promoting stability and equity over time.

## Usage

This package can be integrated into federated learning workflows to:

Quantify fairness across client updates over time.
Implement client selection strategies that enhance fairness in CFL settings.

##Installation

1. Install some of required packages. You can also install them manually.

```
pip install -r requirements.txt # install requirements
```

2. Download the processed dataset here: [Google Drive](https://drive.google.com/file/d/1F7li0NbFWbdaMsqpGUGevEYbT8TAsAx3/view?usp=share_link).
   Unzip this file and place it in the root directory.

3. To run our models:

```
sh run-ous.sh # run generative replay based models
```


## Note

In the generative replay module, for the generative model, we adopt the commonly used backbone generative model (WGAN) in [Deep Generative Replay](https://github.com/kuc2477/pytorch-deep-generative-replay) (DGR). As specified in the paper, the backbone AC-GAN in our FedCIL is a tiny model with a similar structure to the WGAN in the DGR implementation.

The difference is that we add an auxiliary classification head to the top of the discriminator, as introduced in [AC-GAN](https://arxiv.org/pdf/1610.09585.pdf).

## Additional Information

The original `serverbase` model includes a random selection technique. If you need to work with a specific client selection technique, simply use the modified `serverbase` provided.
