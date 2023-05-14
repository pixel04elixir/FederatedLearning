from FLearning import FLSniper

img_path = "./archive/trainingSet/trainingSet"
fl = FLSniper(img_path, rounds=10, poison_clients=0.4, momentum=0.5)
fl.initialize()
fl.train()
