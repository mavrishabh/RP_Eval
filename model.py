# %%
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Conv2DTranspose, Concatenate, Input, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG19, ResNet50V2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

#%%
def recurrent_residual_conv(input_tensor, filters, t=2):
    x = input_tensor
    for _ in range(t):
        if _ == 0:
            x1 = layers.Conv2D(filters, (3, 3), padding="same", use_bias=False)(x)
            x1 = layers.BatchNormalization()(x1)
            x1 = layers.ReLU()(x1)
        else:
            x1 = layers.Conv2D(filters, (3, 3), padding="same", use_bias=False)(x + x1)
            x1 = layers.BatchNormalization()(x1)
            x1 = layers.ReLU()(x1)
    return x1

def r2_block(input_tensor, filters, t=2):
    """Recurrent residual block for R2U-Net."""
    # Adjust input tensor depth if it does not match the desired filter size
    if input_tensor.shape[-1] != filters:
        input_tensor = layers.Conv2D(filters, (1, 1), padding="same")(input_tensor)
    
    x1 = recurrent_residual_conv(input_tensor, filters, t=t)
    return layers.Add()([input_tensor, x1])

# %%
def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

# %%
def decoder_block(inputs, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x
    
# %%
def FFM(low_level_features, high_level_features, num_filters):
    # Upsampling
    high_level_upsampled = UpSampling2D(size=(2, 2))(high_level_features)
    high_level_upsampled = Conv2D(num_filters, (3, 3), padding="same")(high_level_upsampled)
    high_level_upsampled = BatchNormalization()(high_level_upsampled)
    high_level_upsampled = Activation("relu")(high_level_upsampled)

    # Perform 1x1 convolution on low-level features
    low_level_processed = Conv2D(num_filters, (1, 1), padding="same")(low_level_features)
    low_level_processed = BatchNormalization()(low_level_processed)
    low_level_processed = Activation("relu")(low_level_processed)

    # Combine low-level and high-level features
    combined = Concatenate()([low_level_processed, high_level_upsampled])
    
    # Fuse features
    fused = Conv2D(num_filters, (1, 1), padding="same")(combined)
    fused = BatchNormalization()(fused)
    fused = Activation("relu")(fused)

    return fused
    
# %%
def build_vgg19_unet(input_shape):
    inputs = layers.Input(input_shape)

    # Load VGG19
    vgg19 = VGG19(include_top=False, weights="imagenet", input_tensor=inputs)
    vgg_s1 = vgg19.get_layer("block1_conv2").output         # (512, 512, 64)
    vgg_s2 = vgg19.get_layer("block2_conv2").output         # (256, 256, 128)
    vgg_s3 = vgg19.get_layer("block3_conv4").output         # (128, 128, 256)
    vgg_s4 = vgg19.get_layer("block4_conv4").output         # (64, 64, 512)
    vgg_s5 = vgg19.get_layer("block5_conv4").output         # (32, 32, 512)

    # R2U-Net Feature Extraction
    r2u_s1 = r2_block(layers.Conv2D(64, (1, 1), padding="same")(inputs), 64, t=2)
    r2u_s2 = r2_block(layers.MaxPooling2D((2, 2))(r2u_s1), 128, t=2)
    r2u_s3 = r2_block(layers.MaxPooling2D((2, 2))(r2u_s2), 256, t=2)
    r2u_s4 = r2_block(layers.MaxPooling2D((2, 2))(r2u_s3), 512, t=2)
    r2u_s5 = r2_block(layers.MaxPooling2D((2, 2))(r2u_s4), 512, t=2)

    # Combine corresponding layers from R2U-Net and ResNetV2
    s1 = layers.Conv2D(64, (1, 1), padding="same")(layers.Concatenate()([r2u_s1, vgg_s1]))
    s2 = layers.Conv2D(128, (1, 1), padding="same")(layers.Concatenate()([r2u_s2, vgg_s2]))
    s3 = layers.Conv2D(256, (1, 1), padding="same")(layers.Concatenate()([r2u_s3, vgg_s3]))
    s4 = layers.Conv2D(512, (1, 1), padding="same")(layers.Concatenate()([r2u_s4, vgg_s4]))
    s5 = layers.Conv2D(512, (1, 1), padding="same")(layers.Concatenate()([r2u_s5, vgg_s5]))
    
    # Apply FFM on each pair of layers
    f1 = FFM(s1, s2, 64)
    f2 = FFM(s2, s3, 128)
    f3 = FFM(s3, s4, 256)
    f4 = FFM(s4, s5, 512)
 
    # Decoder with upsampling
    d1 = decoder_block(s5, f4, 512)
    d2 = decoder_block(d1, f3, 256)
    d3 = decoder_block(d2, f2, 128)
    
    # Final upsampling and combination
    x1 = UpSampling2D(size=(2, 2))(d3)
    combined = Concatenate()([x1, f1])
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(combined)
 
    # Create model
    model = Model(inputs, outputs, name="R2U-ResNet-UNet")
    return model
    
# %%
if __name__ == "__main__":
    input_shape = (512, 512, 3)
    model = build_vgg19_unet(input_shape)
    model.summary()


