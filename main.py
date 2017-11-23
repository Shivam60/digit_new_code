import collect,network,activations,dev_test,os,pickle
#t, v, test_data=collect.load_mnist()
path="/home/shivam/Work/Projects/JPG-PNG-to-MNIST-NN-Format-master/"
folder_name='test-images'
print("Processing Data: ")
if(folder_name+'_train' in os.listdir() and folder_name+'_valid' in os.listdir() ):
	fl=open('Test_train','rb')
	t=pickle.load(fl)
	fl=open('Test_valid','rb')
	v=pickle.load(fl)
	fl.close()
else:
	t,v=dev_test.get_data(path,folder_name)
print("Data Import Complete")
nn=network.perceptron([28*28,17,46],epochs=10000)
nn.fit(t,v)

