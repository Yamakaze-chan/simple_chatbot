# simple_chatbot
This is just a simple chatbot train with CNN   

model=tf.keras.Sequential()   
model.add(tf.keras.layers.Dense(128,activation='relu',input_shape=(len(X_train[0]),)))   
model.add(tf.keras.layers.Dropout(0.5))   
model.add(tf.keras.layers.Dense(64,activation='relu'))   
model.add(tf.keras.layers.Dense(64,activation='relu'))   
model.add(tf.keras.layers.Dropout(0.5))   
model.add(tf.keras.layers.Dense(len(y_train[0]),activation='softmax'))   
   
You can modify your question and answer of your bot by modify intents.json of run self_learning_chatbot.py   
Train your bot with file train_chatbot.py   
Run your bot with command "python chatbot.py"   
