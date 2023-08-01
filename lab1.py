import numpy
import numpy as np
import matplotlib.pyplot as plt
######### data table ###########
x_train = np.array([1,2])
y_train = np.array([300,500])
print(f"x_train = {x_train}")
print(f"y_train = {y_train}")
########number of training examples m #######
m = x_train.shape
print(f"m_train set = {m}")
print(len(x_train))
m = len(x_train)
print(f"m_train set = {m}")

#######training example x_i , y_i#######
i = 0
x_i = x_train[i]
y_i = y_train[i]
print(f"(x^({i})),(y^({i})= ({x_i},{y_i})")

#######plotting tha data#######
#plot the data point
plt.scatter(x_train,y_train,marker='x',c='b')
#set the title
plt.title("Housing price")
#set y_axis label
plt.ylabel('price (in 1000s of dollar)')
#set x_axis label
plt.xlabel('size(100 sqft)')
plt.show()

#######real example of unvariante regrission #######
# formula is f = wx+b
w = 200
b = 100
print(f"w:{w}")
print(f"b:{b}")
#compute f for x^(0)
f_wb = w * x_train[0] + b
print(f"f_wb:{f_wb}")

# using for

def compute_model_output(x_train,w,b) :
    m = x_train.shape[0]
    f_wb = np.zeros(m)
    for i in range(m) :
        f_wb[i] = w * x_train[i] +b
    return f_wb
tmp_f_wb = compute_model_output(x_train,w,b)

###plot our model prediction
plt.plot(x_train,tmp_f_wb,c='b',label='our prediction')
###plot the data points
plt.scatter(x_train,y_train,marker='x',c='r',label='A')
#set the title
plt.title("Housing price")
#set y_axis label
plt.ylabel('price (in 1000s of dollar)')
#set x_axis label
plt.xlabel('size(100 sqft)')
plt.legend()
plt.show()


### Prediction
# Now that we have a model,
# we can use it to make our original prediction.
# Let's predict the price of a house with 1200 sqft.
# Since the units of $x$ are in 1000's of sqft, $x$ is 1.2.

w = 200
b = 100
x_i = 1.2
cost_1200sqft = w * x_i + b

print(f"${cost_1200sqft:.0f} thousand dollars")



