import matplotlib.pyplot as plt

from FLearning import FLBase
from FLearning import FLPoison
from FLearning import FLSniper

img_path = "./archive/trainingSet/trainingSet"
# fl1 = FLPoison(img_path, momentum=0)
fl2 = FLPoison(img_path, momentum=0)
x, y1, y2 = [], [], []
for i in range(0, 6):
    # fl1.initialize(poison_clients=i/10)
    fl2.initialize(poison_clients=i / 10)
    # fl1.train(rounds=5)
    fl2.train(rounds=5)
    # acc1, loss1 = fl1.evaluate()
    acc2, loss2 = fl2.evaluate()
    x.append(i * 10)
    # y1.append(acc1*100)
    y2.append(acc2 * 100)

# plt.plot(x, y1)
# plt.xlabel("Number of Attackers (%)")
# plt.ylabel("Global Accuracy (%)")
# plt.title("Accuracy vs Number of Attackers")
# plt.legend()
# plt.show()

plt.plot(x, y2)
plt.xlabel("Number of Attackers (%)")
plt.ylabel("Global Accuracy (%)")
plt.title("Accuracy vs Number of Attackers")
plt.legend()
plt.show()

# plt.plot(x, y1, label="Poison")
# plt.plot(x, y2, label="Sniper")
# plt.xlabel("Number of Attackers (%)")
# plt.ylabel("Global Accuracy (%)")
# plt.title("Accuracy vs Number of Attackers")
# plt.legend()
# plt.show()
