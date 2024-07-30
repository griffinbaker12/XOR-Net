def f(x):
    return x**2 + 2 * x + 1


def f_prime(x):
    return 2 * x + 2


def step_towards_minimum(old_x, i, lr):
    grad = f_prime(old_x)
    if grad < 0.0001:
        print(f"Found minimum at: {old_x:.4f}")
        exit(0)
    x_new = old_x - lr * grad
    print(f"Iteration {i+1}: x = {x_new:.4f}, f(x) = {f(x_new):.4f}")
    return x_new


def gradient_descent(start_x, learning_rate):
    x = start_x
    i = 1
    while True:
        new_x = step_towards_minimum(x, i, learning_rate)
        x = new_x
        i += 1


start_x = 2
learning_rate = 0.1
gradient_descent(start_x, learning_rate)