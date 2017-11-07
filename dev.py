import numpy as np
fl={"0": "digit_4", "1": "digit_0", "2": "digit_6", "3": "digit_8", "4": "digit_2", "5": "digit_3", "6": "digit_7", "7": "digit_5", "8": "digit_1", "9": "digit_9"}
d={
	'digit_0':np.array([1,0,0,0,0,0,0,0,0,0]),
	'digit_1':np.array([0,1,0,0,0,0,0,0,0,0]),
	'digit_2':np.array([0,0,1,0,0,0,0,0,0,0]),
	'digit_3':np.array([0,0,0,1,0,0,0,0,0,0]),
	'digit_4':np.array([0,0,0,0,1,0,0,0,0,0]),
	'digit_5':np.array([0,0,0,0,0,1,0,0,0,0]),
	'digit_6':np.array([0,0,0,0,0,0,1,0,0,0]),
	'digit_7':np.array([0,0,0,0,0,0,0,1,0,0]),
	'digit_8':np.array([0,0,0,0,0,0,0,0,1,0]),
	'digit_9':np.array([0,0,0,0,0,0,0,0,0,1])
}
def get_data(path):
	dat_x=np.loadtxt(open(path, "rb"), delimiter=",", skiprows=0)
	dat_y=dat_x[:,0]
	dat_x=np.delete(dat_x,0,1)
	dat_x -= dat_x.min() # normalize the values to bring them into the range 0-1
	dat_x /= dat_x.max()
	dat_y=[str(int(i)) for i in dat_y]
	train=[]
	val=[]
	for i in range(len(dat_x)):
		po=dat_x[i].reshape(32*32,1)
		train.append((po,d[fl[dat_y[i]]].reshape(10,1)))
		val.append((po,fl[dat_y[i]]))
	return train,val
#path="/home/shivam/Work/Projects/sandbox/Datasets/digits_hindi/Test.csv"
#fin=get_data(path)
