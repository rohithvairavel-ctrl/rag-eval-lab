# Backpropagation
Backpropagation computes gradients of the loss w.r.t. each weight using the chain rule, from output layer backward to inputs.
Forward pass computes activations; backward pass propagates error signals. Automatic differentiation frameworks implement this efficiently.
Without backprop, training deep networks by finite differences would be prohibitive.
Common issues: vanishing gradients in deep sigmoid stacks; ReLU, residual links, and batch norm help.
