# KFAC and EKFAC Preconditioners for Pytorch
This repo contains a Pytorch implementation of the KFAC and EKFAC preconditioners.

To use them, first initialize a preconditioner:
```python
preconditioner = KFAC(net, eps=0.1, update_freq=100)
```

Then, simply call the function `step` of the preconditioner before calling the `step` function of the optimizer:

```python
loss.backward()
preconditioner.step()
optimizer.step()
```
