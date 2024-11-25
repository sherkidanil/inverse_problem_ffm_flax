# Inverse problem flow mathcing using flax

## Simple model

```
simple model
|
|---- models/
|---- result_analysis.ipynb
|---- train.py 
```

$$ d(e,m) = e^2 m^3 + m \exp (-\vert 0.2 -e \vert) + \eta $$

## SEIR model

```
simple model
|
|---- models/
|---- generate_dataset.py
|---- understanding_result.ipynb
|---- train.py 
|---- train._with_dataset.py 
```

$$ \frac{dS}{dt} = -\beta(t)SI, \frac{dE}{dt} = \beta(t) S I - \alpha E$$

$$\frac{dI}{dt} = \alpha E - \gamma(t)I, \frac{dR}{dt} = \gamma(t) I$$

 $S(0)=99$, $E(0)=1$, $I(0)=R(0) = 0$

 $$\beta(t) = \beta_1 + \frac{\tanh(7(t-\tau))}{2}(\beta_2 - \beta_1) $$

$$ \gamma(t) = \gamma^r + \gamma^d(t)$$

$$\gamma^d(t) = \gamma^d_1 + \frac{\tanh(7(t-\tau))}{2}(\gamma^d_2 - \gamma^d_1)$$