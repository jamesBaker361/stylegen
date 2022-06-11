import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow.keras.layers import *
import numpy as np

"""It's a convolutional layer that takes in an image and outputs a vector of size `cout`  this is necessary because by default tensorflow convolves starting
from the top left; so the filter anchored at i,j will range from i,j to i+k,j+k, and represent the neighborhood of 
the pixel at i+(k//2),j+(k//2)"""
class ConvOffset(Layer):
    def __init__(self, kernel_size,cout, x,y,padding="same",input_shape=None, *args,**kwargs):
        '''constructor
        
        Parameters
        ----------
        kernel_size
            The size of the kernel to be used in the convolution.
        cout
            number of output channels
        x
            offset from left .
        y
            offset from top.
        padding, optional
            "same" or "valid"
        input_shape
            The shape of the input data (n_H_prev, n_W_prev, n_C_prev)
        
        '''
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
        '''> We create a convolutional layer. We then set the weights of the kernel to be a matrix of zeros, except for
        (for each channel) the element at the position of the desired pixel, which we set to 1
        
        Parameters
        ----------
        input_shape
            the shape of the input tensor
        
        
        '''
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
        '''It takes an input image, then applies the 1 x 1 convolution on the pixel offset from the top left by x,y
        
        Parameters
        ----------
        inputs
            the input image, a tensor of shape [batch_size, height, width, channels]
        
        Returns
        -------
            The output of the dense layer * the output of the conv.
        
        '''
        shape=tf.shape(inputs)
        if len(shape)==3:
            inputs=tf.expand_dims(inputs,0)
            shape=tf.shape(inputs)
        if self.built==False:
            self.build(shape)
        img=self.conv(inputs)
        return self.dense(img)



"""a set of convolutional layers, each with (for each channel) all 0s except for a 1 at one position
followed by a dense layer. ConvBundle(): (B x H x W x C) -> (K x K x B x H x W x C)"""
class ConvBundle(Layer):
    def __init__(self,kernel_size,cout,input_shape=None,*args, **kwargs):
        '''`__init__` is the constructor for the class. initializes the kernel size, the number of output channels, the convolution matrix,
        the dense matrix, and the built variable. If the input shape is not None, it builds the layer.
        
        Parameters
        ----------
        kernel_size
            The size of the kernel.
        cout
            number of output channels
        input_shape
            The shape of the input tensor.
        
        '''
        super().__init__(*args, **kwargs)
        self.kernel_size=kernel_size
        self.cout=cout
        self.conv_matrix=[[None for y in range(kernel_size)] for x in range(kernel_size)]
        self.dense_matrix=[[None for y in range(kernel_size)] for x in range(kernel_size)]
        self.built=False
        if input_shape!=None:
            self.build(input_shape)

    def build(self,input_shape):
        '''For each position in the kernel, we create a convolutional layer with a kernel of size 1x1, and a
        bias of 0. 
        
        The weights of the convolutional layer are set to 1 for the position of the kernel, and 0 for the
        rest. 
        
        The convolutional layer is set to not be trainable. 
        
        The convolutional layer is then added to the conv_matrix; conv_matrix[x][y] is a conv with kernel with all 0s except at index x,y (for each channel)
        
        We also create a dense layer with the same number of outputs as the convolutional layer, and add it
        to the dense_matrix. These function as the weights of the conv matrix
        
        Parameters
        ----------
        input_shape
            The shape of the input tensor.
        
        Returns
        -------
            The output of the convolutional layer.
        
        '''
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
                conv=Conv2D(self.cout,(self.kernel_size,self.kernel_size),padding="same",trainable=False)
                conv.build(input_shape)
                k=np.zeros((self.kernel_size,self.kernel_size,self.cin,self.cout))
                k[x][y]=np.ones((self.cin,self.cout))
                b=np.zeros((self.cout))
                conv.set_weights([k,b])
                self.conv_matrix[x][y]=conv
        self.built=True

    def call(self,inputs):
        '''It takes in a tensor, and returns the tensor if a convolution had been applied but the local values not summed up
        (B x H x W x C) -> (K x K x B x H x W x C)
        
        Parameters
        ----------
        inputs
            The input tensor 
        
        Returns
        -------
            A tensor of shape (kernel_size,kernel_size,batch_size,height,width,channels)
        
        '''
        shape=tf.shape(inputs)
        if self.built is False:
            self.build(shape)
        def _call(inputs):
            output_matrix=[[None for y in range(self.kernel_size)] for x in range(self.kernel_size)]
            for x in range(self.kernel_size):
                for y in range(self.kernel_size):
                    output=self.conv_matrix[x][y](inputs)
                    output=self.dense_matrix[x][y](output)
                    output_matrix[x][y]=output
            return tf.convert_to_tensor(output_matrix)
        if len(shape)==3:
            return _call(tf.expand_dims(inputs,0))
        else:
            return _call(inputs)
        


class SASA(Layer): #Stand-Alone Self Attention
    def __init__(self,kernel_size,cout,use_positional=True,input_shape=None,*args, **kwargs):
        '''constructor
        
        Parameters
        ----------
        kernel_size
            The size of the kernel.
        cout
            number of output channels
        use_positional, optional
            Whether to use positional encoding or not.
        input_shape
            The shape of the input tensor.
        
        '''
        super().__init__(*args, **kwargs)
        self.cout=cout
        self.kernel_size=kernel_size
        self.offset_kernel_size=self.kernel_size+1
        self.offset_kernel_size=self.offset_kernel_size //2
        self.offset_index=self.offset_kernel_size-1
        self.use_positional=use_positional
        self.built=False
        if input_shape!=None:
            self.build(input_shape)

    def build(self, input_shape):
        '''> The `build` function is called when the layer is first used. It creates the `query`, `positional`,
        `values`, and `keys` variables

        
        Parameters
        ----------
        input_shape
            the shape of the input tensor
        '''
        if self.built:
            return
        self.query=ConvOffset(self.offset_kernel_size,self.cout,self.offset_index,self.offset_index)
        if self.use_positional:
            self.positional=tf.Variable(initial_value=tf.random.normal((self.kernel_size,self.kernel_size,self.cout)))
        self.values=ConvBundle(self.kernel_size,self.cout,input_shape=input_shape)
        self.keys=ConvBundle(self.kernel_size,self.cout,input_shape=input_shape)
        self.built=True

    def call(self,inputs):
        '''It takes an input image, and computes self attention 
        
        Parameters
        ----------
        inputs
            The input tensor.
        
        Returns
        -------
            The output of the attention block, same dim as input.
        
        '''
        shape=tf.shape(inputs)
        if len(shape)==3:
            inputs=tf.expand_dims(inputs,0)
        shape=tf.shape(inputs)
        (b,h,w,c)=shape
        inputs=tf.image.pad_to_bounding_box(
            inputs,
            self.offset_index,
            self.offset_index,
            h+self.offset_index,
            w+self.offset_index)
        """we pad the upper left corner with k//2; 
        the query kernel anchored at [i][j] is the same as a 1-D kernel on [i+k//2][j+k//2]
        the keys and value kernels anchored at [i][j] are functions of [i...i+k][j...j+k], and are thus 
        the neighborhood around point [i+k//2][j+k//2] w. radius k//2
        """
        shape=tf.shape(inputs)
        (b,h,w,c)=shape
        if self.built==False:
            self.build(shape)
        
        q=self.query(inputs)
        v=self.values(inputs)
        k=self.keys(inputs)
        kq_product=tf.einsum("pqbijc,bijc->pqbijc",k,q) #multiply q * k
        if self.use_positional:
            pq_product=tf.einsum("pqc,bijc->pqbijc",self.positional,q) #if positional, add positional embeddings
            kq_product=kq_product+pq_product
        c=tf.shape(kq_product)[-1]
        kq_product=tf.reshape(kq_product,(self.kernel_size**2,b*h*w*c))
        kq_product=tf.nn.softmax(kq_product,axis=0) #reshape and softmax
        kq_product=tf.reshape(kq_product,(self.kernel_size,self.kernel_size,b,h,w,c)) #return to original shape
        kqv_product=tf.einsum("pqbijc,pqbijc->pqbijc",v,kq_product)
        product= tf.einsum("pqbijc->bijc",kqv_product) #sum up the weighted attentional area for each pixel
        product=tf.image.crop_to_bounding_box(product,0,0,h-self.offset_index,w-self.offset_index) #the bottom right corner contains the weights for empty white space
        return product

class SASA_MHA(SASA): #multi headed SASA, has a few SASA heads and concatenates the output
    def __init__(self,kernel_size,cout,num_heads,use_positional=True,input_shape=None, *args, **kwargs):
        '''> We create a list of SASA layers, each with the same number of channels
        
        Parameters
        ----------
        kernel_size
            the size of the convolutional kernel.
        cout
            the number of output channels
        num_heads
            the number of heads to use.
        use_positional, optional
            Whether to use positional encoding or not.
        input_shape
            the shape of the input tensor.
        
        '''
        assert cout % num_heads ==0
        self.channels_per_head=cout // num_heads
        self.heads=[SASA(kernel_size,self.channels_per_head,use_positional=use_positional,input_shape=input_shape) for _ in range(num_heads)]
        super().__init__(kernel_size,cout,use_positional,input_shape,*args, **kwargs)

    def build(self, input_shape):
        '''> The function creates a list of heads, and for each head, it calls the build function of the head
        
        Parameters
        ----------
        input_shape
            the shape of the input tensor

        '''
        if self.built:
            return
        for h in self.heads:
            h.build(input_shape)
        self.built=True

    def call(self, inputs):
        '''> The `call` function takes in a tensor `inputs` and returns a tensor `outputs` that is the
        concatenation of the outputs of each head
        
        Parameters
        ----------
        inputs
            the input tensor
        
        Returns
        -------
            The concatenation of the outputs of the heads.
        
        '''
        shape=tf.shape(inputs)
        if self.built==False:
            self.build(shape)
        outputs=[h(inputs) for h in self.heads]
        return tf.concat(outputs,axis=-1)




        
        

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
    sas=SASA_MHA(6,32,4,input_shape=(1,16,16,3))
    sas_out=sas(inputs)
    print(sas_out.shape)