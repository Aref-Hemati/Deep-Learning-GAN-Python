import matplotlib.pyplot as plt
import numpy as np

from keras import backend as K
from keras.datasets import mnist
from keras.layers import Activation,BatchNormalization,Dense,Flatten,\
                         Reshape,Dropout,Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D,Conv2DTranspose
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical

class Dataset:
    def __init__(self,num_labeled):

        self.num_labeled=num_labeled
        (self.x_train, self.y_train), (self.x_test, self.y_test) \
            = mnist.load_data()

        def preprocess_img(x):
            x = (x.astype(np.float)-127.5) / 127.5
            x = np.expand_dims(x,axis=3)
            return x

        def preprocess_label(y):
            return y.reshape(-1,1)

        self.x_train=preprocess_img(self.x_train)
        self.y_train=preprocess_label(self.y_train)
        self.x_test=preprocess_img(self.x_test)
        self.y_test=preprocess_label(self.y_test)

    def read_batch_labeled(self,batch_size):
        ids = np.random.randint(0, self.num_labeled, batch_size)
        imgs = self.x_train[ids]
        labels = self.y_train[ids]
        return imgs,labels

    def read_batch_unlabeled(self,batch_size):
        ids = np.random.randint(self.num_labeled,self.x_train.shape[0], batch_size)
        imgs = self.x_train[ids]
        return imgs

    def read_trainingdata(self):
        x_train = self.x_train[range(self.num_labeled)]
        y_train = self.y_train[range(self.num_labeled)]
        return x_train,y_train

    def read_testingdata(self):
        return self.x_test,self.y_test



num_labeled=100

img_rows=28
img_cols=28
channels=1

dataset = Dataset(num_labeled)

img_shape = (img_rows,img_cols,channels)

zdim=100
num_classes = 10

def build_gen(zdim):

    model = Sequential()
    model.add(Dense(256*7*7,input_dim=zdim))
    model.add(Reshape((7,7,256)))

    #14*14*128
    model.add(Conv2DTranspose(128,kernel_size=3,strides=2,
                              padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))

    # 14*14*64
    model.add(Conv2DTranspose(64,kernel_size=3,strides=1,
                              padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))

    #28*28*1
    model.add(Conv2DTranspose(1,kernel_size=3,strides=2,
                              padding='same'))
    model.add(Activation('tanh'))

    return model

def build_dis(img_shape):

    model=Sequential()

    #14*14*32
    model.add(Conv2D(32,kernel_size=3,strides=2,input_shape=img_shape,
                     padding='same'))
    model.add(LeakyReLU(alpha=0.01))

    #7*7*64
    model.add(Conv2D(64,kernel_size=3,strides=2,input_shape=img_shape,
                     padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))

    #3*3*128
    model.add(Conv2D(128,kernel_size=3,strides=2,input_shape=img_shape,
                     padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(num_classes))
    return model

def build_dis_supervised(dis_net):
    model = Sequential()
    model.add(dis_net)
    model.add(Activation('softmax'))
    return model

def build_dis_unsupervised(dis_net):
    model = Sequential()
    model.add(dis_net)

    def predict(x):
        prediction = 1.0 - (1.0/(K.sum(K.exp(x),axis=-1,keepdims=True)+1.0))
        return prediction

    model.add(Lambda(predict))
    return model

def build_gan(gen,dis):
    model = Sequential()
    model.add(gen)
    model.add(dis)
    return model

dis_v = build_dis(img_shape)

dis_v_sup = build_dis_supervised(dis_v)
dis_v_sup.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

dis_v_unsup = build_dis_unsupervised(dis_v)
dis_v_unsup.compile(loss='binary_crossentropy',
              optimizer=Adam())

gen_v = build_gen(zdim)
dis_v_unsup.trainable=False
gan_v = build_gan(gen_v,dis_v_unsup)
gan_v.compile(loss='binary_crossentropy',
              optimizer=Adam()
             )

supervised_losses=[]
iteration_checks=[]

def train(iterations,batch_size,interval):

    real = np.ones((batch_size,1))
    fake = np.zeros((batch_size, 1))

    for iteration in range(iterations):

        imgs, labels = dataset.read_batch_labeled(batch_size)
        labels = to_categorical(labels,num_classes=num_classes)

        imgs_unlabeled = dataset.read_batch_unlabeled(batch_size)

        z=np.random.normal(0,1,(batch_size,100))
        gen_imgs = gen_v.predict(z)

        dloss_sup, accuracy = dis_v_sup.train_on_batch(imgs,labels)

        dloss_real = dis_v_unsup.train_on_batch(imgs_unlabeled, real)
        dloss_fake = dis_v_unsup.train_on_batch(gen_imgs, fake)

        dloss_unsup = 0.5 * np.add(dloss_real, dloss_fake)

        z = np.random.normal(0, 1, (batch_size, 100))
        gloss = gan_v.train_on_batch(z,real)

        if (iteration+1) % interval == 0:
            supervised_losses.append(dloss_sup)
            iteration_checks.append(iteration+1)

            print("%d [D loss supervised: %.4f , acc: %.2f] [D loss unsupervised: %.4f]" %
                  (iteration+1,dloss_sup,100.0*accuracy,dloss_unsup))

train(3000,32,800)

x, y = dataset.read_trainingdata()
y = to_categorical(y,num_classes=num_classes)

_, accuracy = dis_v_sup.evaluate(x,y)
print("Training Accuracy : %.2f" % (100.0*accuracy))

x, y = dataset.read_testingdata()
y = to_categorical(y,num_classes=num_classes)

_, accuracy = dis_v_sup.evaluate(x,y)
print("Test Accuracy : %.2f" % (100.0*accuracy))

mnist_classifier = build_dis_supervised(build_dis(img_shape))
mnist_classifier.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

imgs, labels = dataset.read_trainingdata()
labels = to_categorical(labels,num_classes=num_classes)

training = mnist_classifier.fit(x=imgs,y=labels,batch_size=32,epochs=30,verbose=1)

x, y = dataset.read_trainingdata()
y = to_categorical(y,num_classes=num_classes)

_, accuracy = mnist_classifier.evaluate(x,y)
print("Training Accuracy : %.2f" % (100.0*accuracy))

x, y = dataset.read_testingdata()
y = to_categorical(y,num_classes=num_classes)

_, accuracy = mnist_classifier.evaluate(x,y)
print("Test Accuracy : %.2f" % (100.0*accuracy))