import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow.keras.layers import *
import numpy as np

class ConvOffset(Layer): #1 x1 convolution offset from the top somehow
    def __init__(self, kernel_size,cout, x,y,padding="same",input_shape=None, *args,**kwargs):
        super().__init__(*args, **kwargs)
        self.kernel_size=kernel_size
        self.x=x
        self.y=y
        self.cout=cout
        self.padding=padding
        if input_shape!=None:
            self.build(input_shape)
        else:
            self.built=False

    def build(self, input_shape):
        if self.built:
            return
        if len(input_shape)==4:
            self.batch_size=input_shape[0]
        else:
            self.batch_size=1
            input_shape=(1, *input_shape)
        self.cin=input_shape[-1]
        self.dense=Dense(self.cout,use_bias=False)
        self.conv=Conv2D(self.cout,(self.kernel_size,self.kernel_size),padding=self.padding,trainable=False)
        self.conv.build(input_shape)
        k=np.zeros((self.kernel_size,self.kernel_size,self.cin,self.cout))
        k[self.x][self.y]=np.ones((self.cin,self.cout))
        b=np.zeros((self.cout))
        self.conv.set_weights([k,b])
        self.built=True

    def call(self, inputs):
        shape=tf.shape(inputs)
        if len(shape)==4:
            (b,h,w,c)=shape
        else:
            b=1
            (h,w,c)=shape
            inputs=tf.expand_dims(inputs,0)
            shape=tf.shape(inputs)
        if self.built==False:
            self.build(shape)
        img=self.conv(inputs)
        return self.dense(img)

            
        



class ConvBundle(Layer):
    def __init__(self,kernel_size,cout,input_shape=None,positional=False,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kernel_size=kernel_size
        self.cout=cout
        self.conv_matrix=[[None for y in range(kernel_size)] for x in range(kernel_size)]
        self.dense_matrix=[[None for y in range(kernel_size)] for x in range(kernel_size)]
        self.positional=positional
        if positional:
            self.positional_matrix=[[None for y in range(kernel_size)] for x in range(kernel_size)]
        if input_shape!=None:
            self.build(input_shape)
        else:
            self.built=False

    def build(self,input_shape):
        if self.built:
            return
        if len(input_shape)==4:
            self.batch_size=input_shape[0]
        else:
            self.batch_size=1
            input_shape=(1, *input_shape)
        self.cin=input_shape[-1]
        for x in range(self.kernel_size):
            for y in range(self.kernel_size):
                d=Dense(self.cout,use_bias=False)
                self.dense_matrix[x][y]=d
                if self.positional:
                    d_p=Dense(self.cout,use_bias=False)
                    self.positional_matrix[x][y]=d_p
                conv=Conv2D(self.cout,(self.kernel_size,self.kernel_size),padding="same",trainable=False)
                conv.build(input_shape)
                k=np.zeros((self.kernel_size,self.kernel_size,self.cin,self.cout))
                k[x][y]=np.ones((self.cin,self.cout))
                b=np.zeros((self.cout))
                conv.set_weights([k,b])
                self.conv_matrix[x][y]=conv
        self.built=True

    def call(self,inputs):
        shape=tf.shape(inputs)
        if self.built is False:
            self.build(shape)
        def _call(inputs):
            output_matrix=[[None for y in range(self.kernel_size)] for x in range(self.kernel_size)]
            for x in range(self.kernel_size):
                for y in range(self.kernel_size):
                    output=self.conv_matrix[x][y](inputs)
                    output=self.dense_matrix[x][y](output)
                    """if self.positional:
                        output+=self.positional_matrix[x][y](inputs)"""
                    output_matrix[x][y]=output
            reshaped=tf.concat(output_matrix,axis=1)
            return tf.convert_to_tensor(output_matrix)
        #return _call(inputs)
        if len(shape)==3:
            return _call(tf.expand_dims(inputs,0))
        else:
            return _call(inputs)
        


class SASA(Layer):
    def __init__(self,kernel_size,cout,input_shape=None,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cout=cout
        self.kernel_size=kernel_size
        if input_shape!=None:
            self.build(input_shape)
        else:
            self.built=False

    def build(self, input_shape):
        if self.built:
            return
        self.offset_kernel_size=self.kernel_size+1
        self.offset_kernel_size=self.offset_kernel_size //2
        self.offset_index=self.offset_kernel_size-1
        self.query=ConvOffset(self.offset_kernel_size,self.cout,self.offset_index,self.offset_index)
        #self.positional=tf.Variable(shape=(self.kernel_size,self.kernel_size,self.cout))
        self.values=ConvBundle(self.kernel_size,self.cout,input_shape=input_shape)
        self.keys=ConvBundle(self.kernel_size,self.cout,input_shape=input_shape)
        self.built=True

    def call(self,inputs):
        shape=tf.shape(inputs)
        if len(shape)==3:
            inputs=tf.expand_dims(inputs,0)
        if self.built==False:
            self.build(shape)
        shape=tf.shape(inputs)
        (b,h,w,c)=shape
        inputs=tf.image.pad_to_bounding_box(
            inputs,
            self.offset_index,
            self.offset_index,
            h+self.offset_index,
            w+self.offset_index)
        shape=tf.shape(inputs)
        (b,h,w,c)=shape
        q=self.query(inputs)
        v=self.values(inputs)
        k=self.keys(inputs)
        kq_product=tf.einsum("pqbijc,bijc->pqbijc",k,q)
        c=tf.shape(kq_product)[-1]
        kq_product=tf.reshape(kq_product,(self.kernel_size**2,b*h*w*c))
        kq_product=tf.nn.softmax(kq_product,axis=0)
        kq_product=tf.reshape(kq_product,(self.kernel_size,self.kernel_size,b,h,w,c))
        kqv_product=tf.einsum("pqbijc,pqbijc->pqbijc",v,kq_product)
        product= tf.einsum("pqbijc->bijc",kqv_product)
        product=tf.image.crop_to_bounding_box(product,0,0,h-self.offset_index,w-self.offset_index)
        return product

        


        
        

if __name__=="__main__":
    offset=ConvOffset(2,1,1,1)
    inputs=tf.random.uniform((1,16,16,3))
    inputs_1=tf.constant([[[[0.0],[0.0],[0.0]],[[0.0],[0.0],[0.0]],[[0.0],[0.0],[1.0]]]])
    off_ouputs=offset(inputs_1)
    #print(off_ouputs)

    bundle=ConvBundle(4,32)
    inputs=tf.random.uniform((7,16,16,3))
    outputs=bundle(inputs)
    print(outputs.shape)
    sas=SASA(5,32,(1,16,16,3))
    sas_out=sas(inputs)
    print(sas_out.shape)