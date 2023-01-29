import pickle 

model = pickle.load(open('model.pkl', 'rb'))
print(model.predict([[2,0,25,0,1,26,1]]))