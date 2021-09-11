import pickle
filename="finalized_model.pkl"
model=pickle.load(open(filename,"rb"))

def prediction(tweet):
    result=model.predict([tweet])
    return result
while True:
    var = input("Text: ")
    print(prediction(var))
    print("press ctrl+c to stop else give input after Text:")
    
