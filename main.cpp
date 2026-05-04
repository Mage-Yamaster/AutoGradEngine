#include "Operators.hpp"
#include "Optim.hpp"

#include <iostream>
#include <random>

int main() {
    Tensor X({4, 2}, Device::CUDA, false);
    Tensor Y({4, 1}, Device::CUDA, false);

    X.set_data({
        0.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 0.0f,
        1.0f, 1.0f
    });
    Y.set_data({
        0.0f,
        1.0f,
        1.0f,
        0.0f
    });

    Tensor W1({2, 4}, Device::CUDA, true);
    Tensor b1({1, 4}, Device::CUDA, true);
    Tensor W2({4, 1}, Device::CUDA, true);
    Tensor b2({1, 1}, Device::CUDA, true);

    std::mt19937 rng(42);
    std::uniform_real_distribution dist(-1.0f, 1.0f);

    auto init = [&](Tensor& t, float scale) {
        std::vector<float> vals(t.numel());
        for (size_t i = 0; i < t.numel(); ++i) {
            vals[i] = dist(rng) * scale;
        }
        t.set_data(vals);
    };

    init(W1, 0.5f);
    init(b1, 0.1f);
    init(W2, 0.5f);
    init(b2, 0.1f);

    Adam optimizer({W1, b1, W2, b2}, 0.1f);

    constexpr int epochs = 5000;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        Tensor H = relu(matmul(X, W1) + b1);
        Tensor Pred = sigmoid(matmul(H, W2) + b2);
        Tensor Loss = mse_loss(Pred, Y);

        optimizer.zero_grad();
        Loss.backward();
        optimizer.step();

        if (epoch % 100 == 0) {
            std::cout << "epoch " << epoch << " loss: " << Loss.get_data()[0] << std::endl;
        }
    }

    return 0;
}
