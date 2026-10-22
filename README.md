# MNIST CNN Learning Example

Legacy Keras MNIST CNN training example with learning-curve plots and test-set evaluation.

## Overview

The script builds a sequential CNN using convolution, max pooling, dropout, and a softmax classifier. It trains on MNIST, plots training/validation curves, and evaluates the test set.

## Repository contents

- [Code_Jalase_8.py](Code_Jalase_8.py)
- [Jalase 8.pdf](Jalase%208.pdf)
- [LICENSE](LICENSE)

## Requirements

A Python environment with NumPy, Matplotlib, and a compatible legacy Keras backend; access to the MNIST dataset.

## Getting started

Clone this repository and open its directory:

```bash
git clone https://github.com/Ali771025/keras.git
cd keras
```

Inspect the source and the compatibility notes below before setting up an execution environment.

## Compatibility and scope

The example uses older APIs including `Model.predict_classes` and `keras.backend.image_data_format`. A current Keras environment may require migration; exact compatible versions are not pinned. The source header credits Reza. No trained weights or verified accuracy benchmark is supplied.

## Credits and license

This repository contains educational material. Preserve author credits and license notices in the source files. See the license file where provided; inclusion in this repository does not replace existing source attribution.
