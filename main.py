import matplotlib.pyplot as plt

from FLearning import FLPoison
from FLearning import FLSniper

img_path = "./archive/trainingSet/trainingSet"
fl1 = FLPoison(img_path, momentum=0)
fl2 = FLSniper(img_path, momentum=0)
# fl3 = FLSniper(img_path, momentum=0.5)
# fl4 = FLSniper(img_path, momentum=0.9)

y1, y2, y3, y4 = [], [], [], []
for i in range(0, 5):
    fl1.initialize(poison_clients=0.3, flp=i / 5.0)
    fl2.initialize(poison_clients=0.3, flp=i / 5.0)
    # fl3.initialize(poison_clients=i / 10.0)
    # fl4.initialize(poison_clients=i / 10.0)

    y1.append(fl1.train(rounds=10)[-1])
    y2.append(fl2.train(rounds=10)[-1])
    # y3.append(fl3.train(rounds=10)[-1])
    # y4.append(fl4.train(rounds=10)[-1])

x = [i*20 for i in range(6)]

# plt.ylim(60, 100)
# plt.xlim(0, 5)
# plt.plot(x, y1, label="Poisoned FL")
# plt.xlabel("Number of Attackers")
# plt.ylabel("Global Accuracy (%)")
# plt.title("Accuracy vs Number of Attackers")
# plt.legend()
# plt.show()

# plt.plot(x, y2, label="Sniper FL")
# plt.xlabel("Number of Attackers")
# plt.ylabel("Global Accuracy (%)")
# plt.title("Accuracy vs Number of Attackers")
# plt.legend()
# plt.show()

plt.plot(x, y1, label="Poisoned FL")
plt.plot(x, y2, label="Sniper")
# plt.plot(x, y3, label="Sniper + Momentum = 0.5")
# plt.plot(x, y4, label="Sniper + Momentum = 0.9")
plt.xlabel("Poisoned Samples (%)")
plt.ylabel("Global Accuracy (%)")
plt.title("Accuracy vs Number of Poisoned Samples")
plt.legend()
plt.show()
