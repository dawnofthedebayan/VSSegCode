import torch
import torch.nn as nn
# Define the 3D residual block with batch normalization
class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv_res = nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=0)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.conv_res(residual)
        out = self.relu(out)
        return out
    


class ResidualBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv_res = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.conv_res(residual)
        out = self.relu(out)
        return out

# Define the 3D attention block
class AttentionBlock3D(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock3D, self).__init__()
        self.conv_1 = nn.Conv3d(in_channels, in_channels//2, kernel_size=3, padding=0)
        self.conv_2 = nn.Conv3d(in_channels//2, 1, kernel_size=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        g = self.conv_1(x)
        g = self.relu(g)
        g = self.conv_2(g)
        attention = self.sigmoid(g)
        return x * attention + x
    


# Define the 2D attention block
class AttentionBlock2D(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock2D, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels, in_channels//2, kernel_size=3, padding=0)
        self.conv_2 = nn.Conv2d(in_channels//2, 1, kernel_size=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        g = self.conv_1(x)
        g = self.relu(g)
        g = self.conv_2(g)
        attention = self.sigmoid(g)
        return x * attention + x



class AttentionBlock_2(nn.Module):

    def __init__(self, in_channels,concat=False):
        super(AttentionBlock_2, self).__init__()

        
        
        if in_channels == 1:
            self.conv_1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=0)
            self.conv_2 = nn.Conv3d(in_channels, 1, kernel_size=1, padding=1)
        else:
            self.conv_1 = nn.Conv3d(in_channels, in_channels//2, kernel_size=3, padding=0)
            self.conv_2 = nn.Conv3d(in_channels//2, 1, kernel_size=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.concat = concat

    def forward(self, x,x_1):

        
        if self.concat:
            x_1 = torch.cat((x, x_1), dim=1)
        
        
        g = self.conv_1(x_1)
        g = self.relu(g)
        g = self.conv_2(g)
        
        attention = self.sigmoid(g)
        
        return x * attention + x
    

class AttentionBlock_2_2D(nn.Module):

    def __init__(self, in_channels,concat=False):
        super(AttentionBlock_2_2D, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels, in_channels//2, kernel_size=3, padding=0)
        self.conv_2 = nn.Conv2d(in_channels//2, 1, kernel_size=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.concat = concat

    def forward(self, x,x_1):

        
        if self.concat:
            x_1 = torch.cat((x, x_1), dim=1)
        
        
        g = self.conv_1(x_1)
        g = self.relu(g)
        g = self.conv_2(g)
        
        attention = self.sigmoid(g)
        
        return x * attention + x


# Define the U-Net model for 3D images
class UNet25D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet25D, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder_transpose = nn.ModuleList()
        self.decoder_attn = nn.ModuleList()
        self.decoder_residual = nn.ModuleList()
        self.decoder_upsample = nn.ModuleList()
        self.decoder_upsample_2 = nn.ModuleList()
        
        channels = [16, 32, 64, 80, 96]
        upsample_ratio = [16, 8, 4, 2, 1]

        channel_deep_sup = [8,16,32]
        # Encoder
        for index,channel in enumerate(channels):
            if index <= 1:

                self.encoder.append(nn.Sequential(
                    nn.Conv2d(in_channels, channel, kernel_size=3, padding=1),
                    nn.BatchNorm2d(channel),
                    nn.ReLU(inplace=True),
                    ResidualBlock2D(channel, channel),
                    nn.MaxPool2d(kernel_size=2, stride=2)
                ))

            else:

                self.encoder.append(nn.Sequential(
                    nn.Conv3d(1, 1, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    ResidualBlock3D(1, 1),
                    nn.MaxPool3d(kernel_size=2, stride=2)
                ))
            in_channels = channel
        
        # Decoder
        for index, channel_tuple in enumerate(zip(reversed(channels), reversed(channels[:-1]))):
            channel_1, channel_2 = channel_tuple

            if index <= 2:
                self.decoder_transpose.append(nn.Sequential(
                    nn.ConvTranspose3d(1, 1, kernel_size=3, stride=2, padding=(1,1,1), output_padding=1),
                    #nn.UpSample(scale_factor=2, mode='trilinear', align_corners=True),
                    nn.ReLU(inplace=True)
                ))

                self.decoder_attn.append(AttentionBlock3D(1 + 1))
                
                self.decoder_residual.append(nn.Sequential(
                    ResidualBlock3D(1 + 1, 1)
                ))

                self.decoder_upsample.append(nn.Sequential(
                    nn.Conv2d(channel_deep_sup[index], out_channels, kernel_size=1, padding=0),
                    nn.Upsample(scale_factor=upsample_ratio[index], mode='bilinear', align_corners=True)
                ))

            else:

                self.decoder_transpose.append(nn.Sequential(
                    nn.ConvTranspose2d(channel_1, channel_1, kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(channel_1),
                    nn.ReLU(inplace=True)
                ))

                self.decoder_attn.append(AttentionBlock2D(channel_1 + channel_2))

                self.decoder_residual.append(nn.Sequential(
                    ResidualBlock2D(channel_1 + channel_2, channel_2)
                ))

                self.decoder_upsample.append(nn.Sequential(
                    nn.Conv2d(channel_2, out_channels, kernel_size=1, padding=0),
                    nn.Upsample(scale_factor=upsample_ratio[index], mode='bilinear', align_corners=True)
                ))



        
        # Final convolutional layer
        self.final_conv = nn.Sequential(
            nn.Conv2d(channels[0], out_channels, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

    def forward(self, x):
        skip_connections = []
        decoder_maps = []
        
        # Encoder
        for index, encoder_block in enumerate(self.encoder):
            if index == 2:
                # expand dimension of x for 3D convolution
                x = x.unsqueeze(1)
            x = encoder_block(x)
            skip_connections.append(x)

            

        # Decoder
        for index, block in enumerate(zip(self.decoder_transpose, self.decoder_attn, self.decoder_residual, self.decoder_upsample)):
            transpose_block, attn_block, res_block, upsample_block = block

            if index == 0:

                
                
                x = transpose_block(x)

                skip = skip_connections[len(skip_connections) - 2 - index]

                
                x = torch.cat((x, skip), dim=1)
                x = attn_block(x)
                x = res_block(x)
                x_upsample = x.squeeze(1)
                decoder_maps.append(upsample_block(x_upsample))
            else:

                skip = skip_connections[len(skip_connections) - 2 - index]
                if index == 3:
                    x = x.squeeze(1)
                x = transpose_block(x)
                

                if index  == 2:

                    skip = skip.unsqueeze(1)
                  
                x = torch.cat((x, skip), dim=1)
                x = attn_block(x)
                x = res_block(x)
                if index <= 2:
                    x_upsample = x.squeeze(1)
                    decoder_maps.append(upsample_block(x_upsample))
                else:
                    decoder_maps.append(upsample_block(x))
        
        # Final convolution
        x = self.final_conv(x)
        
        return x, decoder_maps, skip_connections
    



# Define a DoubleUNet model for 3D images
class DoubleUNet25D(nn.Module):
    def __init__(self, in_channels, out_channels, concat_features_encoder=False, encoder_attention=1, model_1=None):
        super(DoubleUNet25D, self).__init__()
        self.model_1 = model_1
        self.in_channels = in_channels

        if model_1 is None:
            self.model_1 = UNet25D(in_channels - 1, out_channels)
        else:
            print("Model 1 loaded successfully from checkpoint")

        self.encoder = nn.ModuleList()
        self.decoder_transpose = nn.ModuleList()
        self.decoder_attn = nn.ModuleList()
        self.decoder_residual = nn.ModuleList()
        self.decoder_upsample = nn.ModuleList()
        self.encoder_attn = nn.ModuleList()
        self.encoder_attention = encoder_attention
        self.concat_features_encoder = concat_features_encoder
        self.sigmoid = nn.Sigmoid()


        channels = [16, 32, 64, 80, 96]
        upsample_ratio = [16, 8, 4, 2, 1]

        
        if self.encoder_attention == 1:
            for index, channel in enumerate(channels):

                if index <= 1:
                    self.encoder_attn.append(AttentionBlock_2_2D(channel))
                else:
                    self.encoder_attn.append(AttentionBlock_2(1))

        else: 
            for index, channel in enumerate(channels):
                
                if index <= 1:
                    self.encoder_attn.append(AttentionBlock_2_2D(channel*2,concat=True))

                else:

                    self.encoder_attn.append(AttentionBlock_2(2,concat=True))

        if concat_features_encoder:
            channels_enc = [in_channels, 16, 32, 64, 80, 96]
            channel_deep_sup = [8,16,32]
            for index, channel in enumerate(channels_enc[:-1]):
                if index == 0:
                    self.encoder.append(nn.Sequential(
                        nn.Conv2d(channels_enc[index], channels_enc[index + 1], kernel_size=3, padding=1),
                        nn.BatchNorm3d(channels_enc[index + 1]),
                        nn.ReLU(inplace=True),
                        ResidualBlock2D(channels_enc[index + 1], channels_enc[index + 1]),
                        nn.MaxPool2d(kernel_size=2, stride=2)
                    ))

                elif index == 1 or index == 2:

                    self.encoder.append(nn.Sequential(
                        nn.Conv2d(channels_enc[index] * 2, channels_enc[index + 1], kernel_size=3, padding=1),
                        nn.BatchNorm2d(channels_enc[index + 1]),
                        nn.ReLU(inplace=True),
                        ResidualBlock2D(channels_enc[index + 1], channels_enc[index + 1]),
                        nn.MaxPool2d(kernel_size=2, stride=2)
                    ))


                else:
                    self.encoder.append(nn.Sequential(
                        nn.Conv3d(channels_enc[index] * 2, channels_enc[index + 1], kernel_size=3, padding=1),
                        nn.BatchNorm3d(channels_enc[index + 1]),
                        nn.ReLU(inplace=True),
                        ResidualBlock3D(channels_enc[index + 1], channels_enc[index + 1]),
                        nn.MaxPool3d(kernel_size=2, stride=2)
                    ))
        else:
            for index, channel in enumerate(channels):
                
                if index <= 1: 
                
                    self.encoder.append(nn.Sequential(
                        nn.Conv2d(in_channels, channel, kernel_size=3, padding=1),
                        nn.BatchNorm2d(channel),
                        nn.ReLU(inplace=True),
                        ResidualBlock2D(channel, channel),
                        nn.MaxPool2d(kernel_size=2, stride=2)
                    ))

                else:

                    self.encoder.append(nn.Sequential(
                        nn.Conv3d(1, 1, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                        ResidualBlock3D(1, 1),
                        nn.MaxPool3d(kernel_size=2, stride=2)
                    ))
                in_channels = channel

        for index, channel_tuple in enumerate(zip(reversed(channels), reversed(channels[:-1]))):
            channel_1, channel_2 = channel_tuple
            channel_deep_sup = [8,16,32]
            if index <= 2:

                self.decoder_transpose.append(nn.Sequential(

                    nn.ConvTranspose3d(1, 1, kernel_size=3, stride=2, padding=(1,1,1), output_padding=1),
                    nn.ReLU(inplace=True)
                ))

                self.decoder_attn.append(AttentionBlock3D(2))

                self.decoder_residual.append(nn.Sequential(

                    ResidualBlock3D(1 + 1, 1)
                ))

                self.decoder_upsample.append(nn.Sequential(
                    nn.Conv2d(channel_deep_sup[index], out_channels, kernel_size=1, padding=0),
                    nn.Upsample(scale_factor=upsample_ratio[index], mode='bilinear', align_corners=True)
                ))


            else:

                self.decoder_transpose.append(nn.Sequential(
                    nn.ConvTranspose2d(channel_1, channel_1, kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(channel_1),
                    nn.ReLU(inplace=True)
                ))

                self.decoder_attn.append(AttentionBlock2D(channel_1 + channel_2))

                self.decoder_residual.append(nn.Sequential(
                    ResidualBlock2D(channel_1 + channel_2, channel_2)
                ))

                self.decoder_upsample.append(nn.Sequential(
                    nn.Conv2d(channel_2, out_channels, kernel_size=1, padding=0),
                    nn.Upsample(scale_factor=upsample_ratio[index], mode='bilinear', align_corners=True)
                ))

            

        self.final_conv = nn.Sequential(
            nn.Conv2d(channels[0], out_channels, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

    def forward(self, x):
        skip_connections = []
        decoder_maps = []

        x_1, decoder_maps_1, skip_connections_1 = self.model_1(x[:, :, :, :])
        seg_map = self.sigmoid(x_1)
        seg_map = seg_map[:, :, :, :]

        for index, encoder_block in enumerate(self.encoder):
            if index == 0:
                x = torch.cat((x, seg_map), dim=1)
                x = encoder_block(x)
            else:
                if self.concat_features_encoder:
                    x = torch.cat((x, skip_connections_1[index - 1]), dim=1)
                if self.encoder_attention:

                    x = self.encoder_attn[index - 1](x, skip_connections_1[index - 1])

                if index == 2:
                    x = x.unsqueeze(1)
                x = encoder_block(x)
            skip_connections.append(x)

        for index, block in enumerate(zip(self.decoder_transpose, self.decoder_attn, self.decoder_residual, self.decoder_upsample)):
            transpose_block, attn_block, res_block, upsample_block = block
            if index == 0:
                _ = skip_connections.pop()
                x = transpose_block(x)
                skip = skip_connections[len(skip_connections) - 1 - index]
                x = torch.cat((x, skip), dim=1)

               
                x = attn_block(x)
                x = res_block(x)
                

                

                x_upsample = x.squeeze(1)
                decoder_maps.append(upsample_block(x_upsample))

                
            else:

                skip = skip_connections[len(skip_connections) - 1 - index]
                if index == 3:
                    x = x.squeeze(1)
                x = transpose_block(x)

                if index == 2:
                    skip = skip.unsqueeze(1)
                x = torch.cat((x, skip), dim=1)
                x = attn_block(x)
                x = res_block(x)

                if index <= 2:

                    x_upsample = x.squeeze(1)
                    decoder_maps.append(upsample_block(x_upsample))
                else:
                    decoder_maps.append(upsample_block(x))

        x = self.final_conv(x)

        decoder_maps = decoder_maps + decoder_maps_1

        return x, decoder_maps
               
    
"""
# Example usage for DoubleUNet3D
in_channels = 50
out_channels = 50  # Change to the desired number of output channels
model = UNet25D(in_channels, out_channels)

# Dummy input
x = torch.randn(1, 50, 128, 128)  # Adjust the input shape to your data
output, decoder_maps,_ = model(x)

# Print the output shape
print(f"Output shape model 1: {output.shape}")
for x in decoder_maps:
    print(f"Decoder map shape model 1: {x.shape}")

# Example usage for DoubleUNet3D
in_channels = 100
out_channels = 50  # Change to the desired number of output channels

# Dummy input
x = torch.randn(1, 50, 128, 128)  # Adjust the input shape to your data 
x_1 = torch.randn(1, 50, 128, 128)  # Adjust the input shape to your data

model = DoubleUNet25D(in_channels, out_channels, model_1=model, concat_features_encoder=False, encoder_attention=2)

output, decoder_maps = model(x)

# Print the output shape
print(f"Output shape model 2: {output.shape}")
for x in decoder_maps:
    print(f"Decoder map   model 2: {x.shape}")
"""
