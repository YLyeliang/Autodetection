import matplotlib.pyplot as plt

def plot():
    a=[43,22,18,17.9,17.8,17.877,17.966,17.677,17.878,17.666,17.555,17.877]
    print(len(a))
    b=[54,20,18,16,13,11.9,10.2,10.1,9.77,9.64,8.77,8.44]
    print(len(b))
    x=[i*10 for i in range(12)]

    plt.plot(x,a,'r--',label='without warmup')
    plt.plot(x,b,'g--',label="with warmup")
    plt.xlabel('iterations')
    plt.ylabel("loss")
    plt.legend()
    plt.show()