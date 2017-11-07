import collect,network,activations,dev_test
#t, v, test_data=collect.load_mnist()
t,v=dev_test.get_data("/home/shivam/Work/Projects/sandbox/Datasets/DevanagariHandwrittenCharacterDataset/",'Train')

print("Data Import Complete")
nn=network.NeuralNetwork([28*28,17,46],epochs=10000)
nn.fit(t,v)
