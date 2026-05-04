#pragma once

#include "Tensor.hpp"

Tensor add(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor matmul(const Tensor& a, const Tensor& b);
Tensor relu(const Tensor& a);
Tensor sigmoid(const Tensor& a);
Tensor softmax(const Tensor& a);
Tensor cross_entropy_loss(const Tensor& logits, const Tensor& target);
Tensor mse_loss(const Tensor& pred, const Tensor& target);
Tensor transpose(const Tensor& a, size_t dim0 = 0, size_t dim1 = 1);

Tensor& add_(Tensor& a, const Tensor& b);
Tensor& mul_(Tensor& a, const Tensor& b);
Tensor& relu_(Tensor& a);

Tensor operator+(const Tensor& a, const Tensor& b);
Tensor operator*(const Tensor& a, const Tensor& b);
