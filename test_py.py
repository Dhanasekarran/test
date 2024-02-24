{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "5f9d4378",
   "metadata": {},
   "outputs": [],
   "source": [
    "#simple autoencoder\n",
    "#load data\n",
    "#reshape,normalize\n",
    "#model\n",
    "#summary\n",
    "#compile\n",
    "#fit\n",
    "#predict, result display\n",
    "#loss plot\n",
    "#cifar10\n",
    "import numpy as np\n",
    "import tensorflow as tf\n",
    "from tensorflow.keras.datasets import mnist"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "id": "b5228eda",
   "metadata": {},
   "outputs": [],
   "source": [
    "(x_train,y_train),(x_test,y_test)=mnist.load_data()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "c97e1f54",
   "metadata": {},
   "outputs": [],
   "source": [
    "#normalize\n",
    "x_train=x_train/255.0\n",
    "x_test=x_test/255.0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "id": "dc9f2947",
   "metadata": {},
   "outputs": [],
   "source": [
    "from tensorflow.keras.layers import Dense,Input\n",
    "from tensorflow.keras.models import Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "id": "32f09022",
   "metadata": {},
   "outputs": [],
   "source": [
    "#input,encode,decode\n",
    "input_layer=Input(shape=(28,28))\n",
    "\n",
    "encode=Dense(32,activation='relu')(input_layer)\n",
    "\n",
    "decode=Dense(28,activation='sigmoid')(encode)\n",
    "\n",
    "My_model=Model(input_layer,decode)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "id": "d587b188",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model: \"model_6\"\n",
      "_________________________________________________________________\n",
      " Layer (type)                Output Shape              Param #   \n",
      "=================================================================\n",
      " input_8 (InputLayer)        [(None, 28, 28)]          0         \n",
      "                                                                 \n",
      " dense_45 (Dense)            (None, 28, 32)            928       \n",
      "                                                                 \n",
      " dense_46 (Dense)            (None, 28, 28)            924       \n",
      "                                                                 \n",
      "=================================================================\n",
      "Total params: 1,852\n",
      "Trainable params: 1,852\n",
      "Non-trainable params: 0\n",
      "_________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "My_model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "id": "9e527c6c",
   "metadata": {},
   "outputs": [],
   "source": [
    "My_model.compile(optimizer='adadelta',loss='binary_crossentropy')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "id": "307de370",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/20\n",
      "235/235 [==============================] - 3s 9ms/step - loss: 0.6925 - val_loss: 0.6922\n",
      "Epoch 2/20\n",
      "235/235 [==============================] - 2s 7ms/step - loss: 0.6919 - val_loss: 0.6916\n",
      "Epoch 3/20\n",
      "235/235 [==============================] - 1s 6ms/step - loss: 0.6913 - val_loss: 0.6909\n",
      "Epoch 4/20\n",
      "235/235 [==============================] - 1s 6ms/step - loss: 0.6906 - val_loss: 0.6902\n",
      "Epoch 5/20\n",
      "235/235 [==============================] - 1s 6ms/step - loss: 0.6899 - val_loss: 0.6895\n",
      "Epoch 6/20\n",
      "235/235 [==============================] - 2s 6ms/step - loss: 0.6891 - val_loss: 0.6887\n",
      "Epoch 7/20\n",
      "235/235 [==============================] - 1s 6ms/step - loss: 0.6884 - val_loss: 0.6879\n",
      "Epoch 8/20\n",
      "235/235 [==============================] - 1s 6ms/step - loss: 0.6875 - val_loss: 0.6871\n",
      "Epoch 9/20\n",
      "235/235 [==============================] - 1s 6ms/step - loss: 0.6867 - val_loss: 0.6862\n",
      "Epoch 10/20\n",
      "235/235 [==============================] - 1s 6ms/step - loss: 0.6858 - val_loss: 0.6853\n",
      "Epoch 11/20\n",
      "235/235 [==============================] - 1s 6ms/step - loss: 0.6849 - val_loss: 0.6844\n",
      "Epoch 12/20\n",
      "235/235 [==============================] - 2s 7ms/step - loss: 0.6840 - val_loss: 0.6834\n",
      "Epoch 13/20\n",
      "235/235 [==============================] - 2s 6ms/step - loss: 0.6831 - val_loss: 0.6825\n",
      "Epoch 14/20\n",
      "235/235 [==============================] - 2s 7ms/step - loss: 0.6821 - val_loss: 0.6815\n",
      "Epoch 15/20\n",
      "235/235 [==============================] - 1s 6ms/step - loss: 0.6811 - val_loss: 0.6805\n",
      "Epoch 16/20\n",
      "235/235 [==============================] - 1s 6ms/step - loss: 0.6801 - val_loss: 0.6795\n",
      "Epoch 17/20\n",
      "235/235 [==============================] - 2s 7ms/step - loss: 0.6791 - val_loss: 0.6785\n",
      "Epoch 18/20\n",
      "235/235 [==============================] - 2s 7ms/step - loss: 0.6781 - val_loss: 0.6774\n",
      "Epoch 19/20\n",
      "235/235 [==============================] - 2s 8ms/step - loss: 0.6770 - val_loss: 0.6763\n",
      "Epoch 20/20\n",
      "235/235 [==============================] - 2s 7ms/step - loss: 0.6759 - val_loss: 0.6752\n"
     ]
    }
   ],
   "source": [
    "history=My_model.fit(x_train,x_train, validation_data=(x_test,x_test),batch_size=256,epochs=20,shuffle=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "id": "51f8672c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 648x144 with 5 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "n=5\n",
    "plt.figure(figsize=(9,2))\n",
    "for i in range(n):\n",
    "    ax=plt.subplot(1,n,i+1)\n",
    "    plt.imshow(x_test[i].reshape(28,28),cmap='gray')\n",
    "    ax.axis('off')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "id": "2fbba574",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "313/313 [==============================] - 1s 2ms/step\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 648x144 with 5 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "decoded_images=My_model.predict(x_test)\n",
    "n=5\n",
    "plt.figure(figsize=(9,2))\n",
    "for i in range(n):\n",
    "    ax=plt.subplot(1,n,i+1)\n",
    "    plt.imshow(decoded_images[i].reshape(28,28),cmap='gray')\n",
    "    ax.axis('off')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "id": "b48cd34c",
   "metadata": {},
   "outputs": [],
   "source": [
    "#reshpe for deep autoencoder\n",
    "x_train=np.reshape(x_train,(-1,784))\n",
    "x_test=np.reshape(x_test,(-1,784))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "id": "7fb5e45b",
   "metadata": {},
   "outputs": [],
   "source": [
    "#normalize\n",
    "x_train=x_train/255.0\n",
    "x_test=x_test/255.0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "id": "83f46a9d",
   "metadata": {},
   "outputs": [],
   "source": [
    "input_dim=28*28\n",
    "latent_dim=16"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "id": "ce6e842d",
   "metadata": {},
   "outputs": [],
   "source": [
    "input_layer=Input(shape=(input_dim,))\n",
    "\n",
    "enc_layer1=Dense(500,activation='sigmoid')(input_layer)\n",
    "enc_layer2=Dense(300,activation='sigmoid')(enc_layer1)\n",
    "enc_layer3=Dense(100,activation='sigmoid')(enc_layer2)\n",
    "enc_layer4=Dense(latent_dim,activation='tanh')(enc_layer3)\n",
    "\n",
    "decode1=Dense(100,activation='sigmoid')(enc_layer4)\n",
    "decode2=Dense(300,activation='sigmoid')(decode1)\n",
    "decode3=Dense(500,activation='sigmoid')(decode2)\n",
    "decode4=Dense(input_dim,activation='sigmoid')(decode3)\n",
    "\n",
    "my_deepmodel=Model(input_layer,decode4)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "id": "e909f2df",
   "metadata": {},
   "outputs": [],
   "source": [
    "my_deepmodel.compile(optimizer='adam',loss='binary_crossentropy')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "id": "673cca9e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/2\n",
      "469/469 [==============================] - 14s 25ms/step - loss: 0.0114 - val_loss: 0.0041\n",
      "Epoch 2/2\n",
      "469/469 [==============================] - 11s 24ms/step - loss: 0.0040 - val_loss: 0.0040\n"
     ]
    }
   ],
   "source": [
    "history=my_deepmodel.fit(x_train,x_train,validation_data=(x_test,x_test),epochs=2,batch_size=128,shuffle=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "id": "b53dc02e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 648x144 with 5 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "n=5\n",
    "plt.figure(figsize=(9,2))\n",
    "for i in range(n):\n",
    "    ax=plt.subplot(1,n,i+1)\n",
    "    plt.imshow(x_test[i].reshape(28,28),cmap='gray')\n",
    "    ax.axis('off')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "id": "33d1d7d8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "313/313 [==============================] - 1s 4ms/step\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 648x144 with 5 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "decoded_images=my_deepmodel.predict(x_test)\n",
    "n=5\n",
    "plt.figure(figsize=(9,2))\n",
    "for i in range(n):\n",
    "    ax=plt.subplot(1,n,i+1)\n",
    "    plt.imshow(decoded_images[i].reshape(28,28),cmap='gray')\n",
    "    ax.axis('off')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9f420ec9",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "id": "51b06e1f",
   "metadata": {},
   "outputs": [],
   "source": [
    "#simple \n",
    "#deep\n",
    "#cnn\n",
    "#denoisy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "id": "67ef4663",
   "metadata": {},
   "outputs": [],
   "source": [
    "#conv2d,maxpooling,upsampling\n",
    "from tensorflow.keras.layers import Conv2D,MaxPooling2D,UpSampling2D"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 95,
   "id": "4538f62e",
   "metadata": {},
   "outputs": [],
   "source": [
    "(x_train,y_train),(x_test,y_test)=mnist.load_data()\n",
    "\n",
    "#normalize\n",
    "x_train=x_train/255.0\n",
    "x_test=x_test/255.0\n",
    "\n",
    "x_train=np.clip(x_train,0.,1.)\n",
    "\n",
    "x_test=np.clip(x_test,0.,1.)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 99,
   "id": "69de78ed",
   "metadata": {},
   "outputs": [],
   "source": [
    "#1\n",
    "input_layer=Input(shape=(28,28,1))\n",
    "#encoder\n",
    "x=Conv2D(32,(3,3),activation='relu',padding='same')(input_layer)\n",
    "x=MaxPooling2D((2,2),padding='same')(x)\n",
    "\n",
    "x=Conv2D(32,(3,3),activation='relu',padding='same')(x)\n",
    "encoder=MaxPooling2D((2,2),padding='same')(x)\n",
    "#decoder\n",
    "x=Conv2D(32,(3,3),activation='relu',padding='same')(encoder)\n",
    "x=UpSampling2D((2,2))(x)\n",
    "\n",
    "x=Conv2D(32,(3,3),activation='relu',padding='same')(x)\n",
    "x=UpSampling2D((2,2))(x)\n",
    "\n",
    "decoder=Conv2D(1,(3,3),activation='sigmoid',padding='same')(x)\n",
    "\n",
    "autoencoder=Model(input_layer,decoder)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 100,
   "id": "37ccaf14",
   "metadata": {},
   "outputs": [],
   "source": [
    "autoencoder.compile(optimizer='adam',loss='binary_crossentropy')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 101,
   "id": "634f0eee",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/2\n",
      "235/235 [==============================] - 94s 393ms/step - loss: 0.1542 - val_loss: 0.0835\n",
      "Epoch 2/2\n",
      "235/235 [==============================] - 108s 460ms/step - loss: 0.0799 - val_loss: 0.0758\n"
     ]
    }
   ],
   "source": [
    "history=autoencoder.fit(x_train,x_train,validation_data=(x_test,x_test),epochs=2,batch_size=256)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 102,
   "id": "b9bfcebd",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 648x144 with 5 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "n=5\n",
    "plt.figure(figsize=(9,2))\n",
    "for i in range(n):\n",
    "    ax=plt.subplot(1,n,i+1)\n",
    "    plt.imshow(x_test[i].reshape(28,28),cmap='gray')\n",
    "    ax.axis('off')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 103,
   "id": "3a5332c6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "313/313 [==============================] - 5s 14ms/step\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 648x144 with 5 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "decoded_images=autoencoder.predict(x_test)\n",
    "n=5\n",
    "plt.figure(figsize=(9,2))\n",
    "for i in range(n):\n",
    "    ax=plt.subplot(1,n,i+1)\n",
    "    plt.imshow(decoded_images[i].reshape(28,28),cmap='gray')\n",
    "    ax.axis('off')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 104,
   "id": "bfc1d792",
   "metadata": {},
   "outputs": [],
   "source": [
    "noise_factor=0.5\n",
    "\n",
    "x_train_noisy=x_train+noise_factor*np.random.normal(loc=0.0,scale=1.0,size=x_train.shape)\n",
    "\n",
    "x_test_noisy=x_test+noise_factor*np.random.normal(loc=0.0,scale=1.0,size=x_test.shape)\n",
    "#normalze\n",
    "x_train_noisy=np.clip(x_train_noisy,0.,1.)\n",
    "\n",
    "x_test_noisy=np.clip(x_test_noisy,0.,1.)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 114,
   "id": "df91fa56",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/10\n",
      "235/235 [==============================] - 91s 386ms/step - loss: 0.4861 - val_loss: 0.4835\n",
      "Epoch 2/10\n",
      "235/235 [==============================] - 98s 416ms/step - loss: 0.4812 - val_loss: 0.4777\n",
      "Epoch 3/10\n",
      "235/235 [==============================] - 92s 390ms/step - loss: 0.4743 - val_loss: 0.4700\n",
      "Epoch 4/10\n",
      "235/235 [==============================] - 84s 359ms/step - loss: 0.4667 - val_loss: 0.4624\n",
      "Epoch 5/10\n",
      "235/235 [==============================] - 97s 415ms/step - loss: 0.4596 - val_loss: 0.4564\n",
      "Epoch 6/10\n",
      "235/235 [==============================] - 110s 469ms/step - loss: 0.4545 - val_loss: 0.4519\n",
      "Epoch 7/10\n",
      "235/235 [==============================] - 101s 429ms/step - loss: 0.4502 - val_loss: 0.4481\n",
      "Epoch 8/10\n",
      "235/235 [==============================] - 75s 318ms/step - loss: 0.4461 - val_loss: 0.4440\n",
      "Epoch 9/10\n",
      "235/235 [==============================] - 73s 310ms/step - loss: 0.4424 - val_loss: 0.4401\n",
      "Epoch 10/10\n",
      "235/235 [==============================] - 73s 312ms/step - loss: 0.4388 - val_loss: 0.4364\n"
     ]
    }
   ],
   "source": [
    "history2=autoencoder.fit(x_train_noisy,x_train_noisy,validation_data=(x_test_noisy,x_test_noisy),epochs=10,batch_size=256)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 117,
   "id": "ee116213",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAgQAAABkCAYAAAD5aSjlAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/YYfK9AAAACXBIWXMAAAsTAAALEwEAmpwYAAAntElEQVR4nO2de9zO9f3H3y1EB1YS1XKoaJNtSVZpZdNRB+s0NRFqVpLSYVFIFmZzKKJETmnWlowyI7IcymRkUSYTnZUh0axi/f74Pbx7fj59P9f9ve/70h6P3+/1/Ot13fd1+F7X5/v9XNfjfXi99/n8889NCCGEEP+/+dp/+wCEEEII8d9HPwiEEEIIoR8EQgghhNAPAiGEEEKYfhAIIYQQwswqFPrnPvvsk9mCMGDAANe///3vXVevXj2430MPPeT6vvvuc7127VrXc+fOzXusZmZ2xRVXuD722GNd9+/fP/mYRo0auX7nnXdcb9261fUPfvAD1/vuu6/rZ599NvM5L7zwwuD2jBkzXNeqVcv19773PdetWrVyfe211+6TPOCcVKhQwdfnd7/7nf/98ssvdz1+/HjXP//5z4PHt23b1vU//vEP13wvJ554ouvOnTu7HjZsmOtVq1a57tChg+sjjzzS9eDBg4PX/uSTT770fszMfvjDH7r+85//nHmfZcuWuW7SpEnmfR5++OHg9nXXXZd5P3LEEUe4fuedd8q9Prx+eE7t3r3b9SGHHOJ6y5YtweMrVPji8ty1a5frs846yzWvn/bt27ueOHFiicf31ltvuR4+fHjwv/nz57t+8cUXS3wuctlll7n+5S9/6bpBgwbJx7Rs2dL1Z5995prv75RTTnG9ePHicq9PvXr1fH02bNhQ6sd/85vfdP33v//d9WGHHeb6oosucj127FjXvK6WL1+e+fy8D68lM7Onn37a9SWXXOKanx2v47p167rmez333HNdz549O/M48nLLLbe4Hjp0aLnX55lnnvH1+fGPf+x//+ijj1zznLjzzjuDx//oRz9yzcc/8cQTpTqOs88+2/WcOXNSxxrcPuecczLvt2bNGtfHHXdc5n0eeOAB1127dnXN776XXnopeMzFF1/smufiU089lfkan3/+eeb6KEIghBBCCP0gEEIIIYTZPoWMiVIpgxSHH354cPukk05yPXPmTNcMtb3yyiuuGR5kmPLo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\n",
      "text/plain": [
       "<Figure size 648x144 with 5 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "n=5\n",
    "plt.figure(figsize=(9,2))\n",
    "for i in range(n):\n",
    "    ax=plt.subplot(1,n,i+1)\n",
    "    plt.imshow(x_test_noisy[i].reshape(28,28),cmap='gray')\n",
    "    ax.axis('off')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 118,
   "id": "011f6c3b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "313/313 [==============================] - 3s 11ms/step\n"
     ]
    },
    {
     "data": {
      "image/png": "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e/nLX562mQJKsxa5Gq11115ExJVXXpm2693f1H5PnArFEBQKhUKhUKgXgkKhUCgUCtNIBtIp0jFKAdIYPc00Kj4j7SJ1KDUjLeTn2tJje++9d9O2FJFUlrSszyRlKU0zouVvueWWpj2jr5VEpI4cz0nAYiPSXaP69v1ZCsoJ9vOCCy5Ie911101bytMiM97X6GTHy5rjEe358aeeemraUvjKAbvttlvaniUvrepY90V+HId11lknbalI+/Hud787ZgrnQR9UhvHzfv04fhYRMptAenCXXXZJ2/GWzt58883TtihLL7mZpeGacVylIM1MmTNnTtpG1vt5f/aI6915N3vIdTUJOL7Og/SvEqVZSRERxx9/fNr207H0c8dR35B2d705572c5Dy6JypHOe+25/oZyVc9lO/cN13HrsVJYNSv0W9Rn7FjtpfjJ41uBpttfOpTn0rbondmqSk99G1baEipa9NNN03bzA8lJNe3507MnTs37dVXX71pb3QOhb6lBDNCMQSFQqFQKBTqhaBQKBQKhcI0ksGoAIW2lERfC97/SWncfffdaUvtS2sZFTuqx+zzeRRyRBvdazTnSA6QEpP+8fn8jnRrREu3jWrN94UzJgnblEaWUuuP1/V5pPodOyNejcS2/9JjPodj0tfKly5zfr1ean3nnXdO20h5CyRJL5911llNe9J+Uo4nn3xy2n0k90xhBLS+o0+M5iqiHb9RBo7PbHEUaXepU2lRC5VIQUe0VLlSitHXRrrbJ+daWUqJoT/O1zVnv+3TgijtRYF7kuPl8zvufeE1s0Auv/zytK1jL8VsAS8lNwsCefaBdt+2cqH7pmvX7BDnerQ/2W99N6LNQtFPd9hhhymfYxKwgFgvAc6H+3OfZTCSOZSHvK/y9XHHHZe2lL9t6LMWMopozzZw7Fw/jr1ZW64316u+2MszZra4tvyNGx27LoohKBQKhUKhUC8EhUKhUCgUppEMpGMsNmJBIGmtPtLSI04PPfTQtE866aS0pbWkrKSkpVykvY1wfcc73tG0LX3l8ZPS01KeRrRKrShLSPH0UfMjKsmCRcstt1xMEkoh1uG2EJPSi1GxES1NOsq+0AfM6pDydHylP6W1+qIljoURtu95z3vS9shX+yH96X0sRrXvvvs27UkfSr96X6WPSUDa1QyT1VZbLW0j73vKUz+ULpcWloa3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\n",
      "text/plain": [
       "<Figure size 648x144 with 5 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "decoded_images=autoencoder.predict(x_test_noisy)\n",
    "n=5\n",
    "plt.figure(figsize=(9,2))\n",
    "for i in range(n):\n",
    "    ax=plt.subplot(1,n,i+1)\n",
    "    plt.imshow(decoded_images[i].reshape(28,28),cmap='gray')\n",
    "    ax.axis('off')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 119,
   "id": "5498ed2e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "plt.plot(history2.history['loss'],label='training loss')\n",
    "plt.plot(history2.history['val_loss'],label='validation loss')\n",
    "plt.xlabel('epoch')\n",
    "plt.ylabel('loss')\n",
    "plt.title('loss graph')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f4798b4d",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
