# %%
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Conv2DTranspose, Concatenate, Input, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG19, ResNet50V2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

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
def contraction_path(inputs, dropout_size, num_filters):
    x = tf.keras.layers.Conv2D(num_filters, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    x = tf.keras.layers.Dropout(dropout_size)(x)
    x = tf.keras.layers.Conv2D(num_filters, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
    y = tf.keras.layers.MaxPooling2D((2, 2))(x)
    return y

# %%
def contraction_path_1(inputs, dropout_size, num_filters):
    x = tf.keras.layers.Conv2D(num_filters, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    x = tf.keras.layers.Dropout(dropout_size)(x)
    j = tf.keras.layers.DepthwiseConv2D((3, 3), dilation_rate=2, padding='same', activation='relu')(x)
    k = tf.keras.layers.DepthwiseConv2D((3, 3), dilation_rate=3, padding='same', activation='relu')(x)
    l = tf.keras.layers.DepthwiseConv2D((3, 3), dilation_rate=5, padding='same', activation='relu')(x)
    y = tf.keras.layers.Add()([j, k, l])
    y = tf.keras.layers.MaxPooling2D((2, 2))(y)
    return y

# %%
def expansive_path(inputs, skip_features, dropout_size, num_filters):
    x = tf.keras.layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(inputs)
    x = tf.keras.layers.concatenate([x, skip_features])
    y = tf.keras.layers.Conv2D(num_filters, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
    y = tf.keras.layers.Dropout(dropout_size)(y)
    y = tf.keras.layers.Conv2D(num_filters, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(y) 
    return y

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
# def SIEM(input_features, skip_features, num_filters, slice_size):
#     """
#     Spatial Information Enhancement Module (SIEM)
#     This module enhances the spatial structure of the input feature maps by performing 
#     horizontal and vertical slicing followed by convolution operations.

#     Parameters:
#     - input_features: Feature map from the upsampling path (e.g., x1)
#     - skip_features: Feature map from the skip connection (e.g., f1)
#     - num_filters: Number of filters to use in the convolution layers
#     - slice_size: Slice size (r) for the spatial convolutions.

#     Returns:
#     - siem_output: Enhanced feature map after horizontal and vertical convolutions
#     """
#     # Concatenate input features and skip connection
#     feature_map = Concatenate()([input_features, skip_features])
    
#     # Get the shape of the input feature map (Batch, Height, Width, Channels)
#     _, H, W, C = feature_map.shape
    
#     # ----------------- Horizontal Slice Convolution -----------------
#     # Perform convolution across the horizontal slices (rows)
#     for h in range(H):
#         # Extract each horizontal slice (a row across the channels)
#         horizontal_slice = feature_map[:, h:h+1, :, :]
        
#         # Apply convolution with kernel size (C, slice_size) along the width
#         conv_slice = Conv2D(num_filters, (C, slice_size), padding="same")(horizontal_slice)
        
#         # Accumulate the results into the next slice
#         if h == 0:
#             horizontal_accum = conv_slice
#         else:
#             horizontal_accum = horizontal_accum + conv_slice
    
#     # ----------------- Vertical Slice Convolution -----------------
#     # Perform convolution across the vertical slices (columns)
#     for w in range(W):
#         # Extract each vertical slice (a column across the channels)
#         vertical_slice = feature_map[:, :, w:w+1, :]
        
#         # Apply convolution with kernel size (slice_size, C) along the height
#         conv_slice = Conv2D(num_filters, (slice_size, C), padding="same")(vertical_slice)
        
#         # Accumulate the results into the next slice
#         if w == 0:
#             vertical_accum = conv_slice
#         else:
#             vertical_accum = vertical_accum + conv_slice
    
#     # ----------------- Combine Horizontal and Vertical Convolutions -----------------
#     # Concatenate both the horizontally and vertically processed feature maps
#     siem_output = Concatenate()([horizontal_accum, vertical_accum])
    
#     return siem_output

# %%
def build_vgg19_unet(input_shape):
    inputs = Input(input_shape)

    #  # Load VGG19 and ResNetV2
    vgg19 = VGG19(include_top=False, weights="imagenet", input_tensor=inputs)
    resnet = ResNet50V2(include_top=False, weights="imagenet", input_tensor=inputs)
    
    # Extract feature layers from VGG19
    vgg_s1 = vgg19.get_layer("block1_conv2").output         # (512, 512, 64)
    vgg_s2 = vgg19.get_layer("block2_conv2").output         # (256, 256, 128)
    print(vgg_s2.shape)
    vgg_s3 = vgg19.get_layer("block3_conv4").output         # (128, 128, 256)
    print(vgg_s3.shape)
    vgg_s4 = vgg19.get_layer("block4_conv4").output         # (64, 64, 512)
    print(vgg_s4.shape)
    vgg_s5 = vgg19.get_layer("block5_conv4").output         # (32, 32, 512)
    print(vgg_s5.shape)

    # Extract feature layers from ResNetV2 and resize to match VGG19 layer sizes
    res_s1 = Conv2D(64, (1, 1), padding="same")(UpSampling2D(size=(2, 2))(resnet.get_layer("conv1_conv").output))           # Up to (512, 512)
    res_s2 = Conv2D(128, (1, 1), padding="same")(UpSampling2D(size=(4, 4))(resnet.get_layer("conv2_block3_out").output))    # Up to (256, 256)
    res_s3 = Conv2D(256, (1, 1), padding="same")(UpSampling2D(size=(4, 4))(resnet.get_layer("conv3_block4_out").output))    # Up to (128, 128)
    res_s4 = Conv2D(512, (1, 1), padding="same")(UpSampling2D(size=(4, 4))(resnet.get_layer("conv4_block6_out").output))    # Up to (64, 64)
    res_s5 = Conv2D(512, (1, 1), padding="same")(UpSampling2D(size=(2, 2))(resnet.get_layer("conv5_block3_out").output))    # Up to (32, 32)

    # Combine corresponding layers from VGG19 and ResNetV2 using a convolution layer
    s1 = Conv2D(64, (1, 1), padding="same")(Concatenate()([vgg_s1, res_s1]))
    s2 = Conv2D(128, (1, 1), padding="same")(Concatenate()([vgg_s2, res_s2]))
    s3 = Conv2D(256, (1, 1), padding="same")(Concatenate()([vgg_s3, res_s3]))
    s4 = Conv2D(512, (1, 1), padding="same")(Concatenate()([vgg_s4, res_s4]))
    s5 = Conv2D(512, (1, 1), padding="same")(Concatenate()([vgg_s5, res_s5]))
    
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
    model = Model(inputs, outputs, name="VGG19_ResNet_UNet")
    return model
    # inputs = tf.keras.layers.Input(input_shape)
    # s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)
    
    # #Contraction path 1
    # s1 = (UpSampling2D(size = (2,2)))(contraction_path(s, 0.1, 32))
    # s2 = contraction_path(s1, 0.1, 64)
    # s3 = contraction_path(s2, 0.2, 128)
    # s4 = contraction_path(s3, 0.2, 256)
    # s5 = contraction_path(s4, 0.3, 256)
    
    # #Contraction path 2
    # c1 = (UpSampling2D(size = (2,2)))(contraction_path_1(s, 0.1, 32))
    # c1 = tf.keras.layers.Add()([c1, s1])
    # print(c1.shape)
    # c2 = contraction_path_1(s1, 0.1, 64)
    # c2 = tf.keras.layers.Add()([c2, s2])
    # print(c2.shape)
    # c3 = contraction_path_1(s2, 0.2, 128)
    # c3 = tf.keras.layers.Add()([c3, s3])
    # print(c3.shape)
    # c4 = contraction_path_1(s3, 0.2, 256)
    # c4 = tf.keras.layers.Add()([c4, s4])
    # print(c4.shape)
    # c5 = contraction_path_1(s4, 0.3, 256)
    # c5 = tf.keras.layers.Add()([c5, s5])
    # print(c5.shape)
    
    # #Expansive Path
    # u1 = expansive_path(c5, c4, 0.2, 256)
    # print(u1.shape)
    # u2 = expansive_path(u1, c3, 0.2, 128)
    # print(u2.shape)
    # u3 = expansive_path(u2, c2, 0.1, 64)
    # print(u3.shape)
    # u4 = expansive_path(u3, c1, 0.1, 32)
    # print(u4.shape)
    
    
    # # Apply FFM on each pair of layers
    # f1 = FFM(s1, u4, 32)
    # f2 = FFM(s2, u3, 64)
    # f3 = FFM(s3, u2, 128)
    # f4 = FFM(s4, u1, 256)

    # # Decoder with upsampling
    # d1 = decoder_block(s5, f4, 256)
    # d2 = decoder_block(d1, f3, 128)
    # d3 = decoder_block(d2, f2, 64)
    
    # x1 = UpSampling2D(size=(2, 2))(d3)
    # # siem = SIEM(x1, f1, 64, 2)
    # combined = Concatenate()([x1, f1])
    # outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(combined)

    # model = Model(inputs, outputs, name="VGG19_U-Net")
    # return model

# %%
if __name__ == "__main__":
    input_shape = (512, 512, 3)
    model = build_vgg19_unet(input_shape)
    model.summary()


