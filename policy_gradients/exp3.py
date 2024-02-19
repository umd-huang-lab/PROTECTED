import numpy as np
from matplotlib import pyplot as plt


class Exp3Alg:
    def __init__(self, K, T, normalizer, lr=2.0):
        self.K = K
        self.T = T
        self.w = np.array([1.] * K)
        self.eta = lr * np.sqrt(np.log(K) / (K * T))
        self.gamma = self.eta / 2
        self.normalizer = normalizer

    def sample(self):
        p = self.w / self.w.sum()
        return np.random.choice(a=self.K, p=p)

    def get_policy(self):
        return self.w / self.w.sum()

    def update(self, r, i):
        p = self.w / self.w.sum()
        p_i = p[i]
        r = r / self.normalizer
        r_hat = r / (p_i + self.gamma)
        self.w[i] = self.w[i] * np.exp(self.eta * r_hat)

    def reset(self):
        self.w = np.array([1] * self.K)


def main():
    K = 10
    T = 2441
    game = np.random.random(size=(K, K))
    game = 1 - game
    exp1 = Exp3Alg(K, T, 1)
    exp2 = Exp3Alg(K, T, 1)
    reward1_list = []
    reward2_list = []

    exp_reward1_list = []
    exp_reward2_list = []

    reward_vec1_list = []
    reward_vec2_list = []

    for i in range(T):
        a_1 = exp1.sample()
        exp2.reset()
        a_2 = exp2.sample()
        r_1 = game[a_1, a_2]
        r_2 = 1 - r_1
        reward1_list.append(r_1)
        reward2_list.append(r_1)
        reward_vec1_list.append((game @ exp2.get_policy().reshape(-1, 1)).flatten())
        reward_vec2_list.append(1 - (exp1.get_policy().reshape(1, -1) @ game).flatten())
        exp_reward1_list.append((exp1.get_policy() * reward_vec1_list[-1]).sum())
        exp_reward2_list.append((exp2.get_policy() * reward_vec2_list[-1]).sum())
        if i % 1 == 0:
            print(f"reward vector is {reward_vec1_list[-1]}")
            # print(f"w is {exp1.w}")
            print(f"policy is {exp1.get_policy()}")
        exp1.update(r_1, a_1)
        exp2.update(r_2, a_2)

    regret1_list = []
    regret2_list = []
    for i in range(T):
        regret1_list.append((np.sum(reward_vec1_list[0:i + 1]).max() - sum(exp_reward1_list[0:i + 1])) / (i + 1))
        regret2_list.append((np.sum(reward_vec2_list[0:i + 1]).max() - sum(exp_reward2_list[0:i + 1])) / (i + 1))
    plt.plot(regret1_list, label="p1")
    plt.plot(regret2_list, label="p2")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    for _ in range(1):
        main()
