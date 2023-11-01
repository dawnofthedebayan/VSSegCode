import torch
import torch.nn as nn

# Define the residual block with batch normalization
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
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

# Define the attention block
class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
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

# attention of previous encoder and current encoder 
class AttentionBlock_2(nn.Module):

    def __init__(self, in_channels):
        super(AttentionBlock_2, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels*2, (in_channels*2)//2, kernel_size=3, padding=0)
        self.conv_2 = nn.Conv2d((in_channels*2)//2, 1, kernel_size=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x,x_1):
        
        x_c = torch.cat((x, x_1), dim=1)
        g = self.conv_1(x_c)
        g = self.relu(g)
        g = self.conv_2(g)
        
        attention = self.sigmoid(g)
        
        return x * attention + x

    

class AttentionBlock_3(nn.Module):

    def __init__(self, in_channels,concat=False):
        super(AttentionBlock_3, self).__init__()
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
    





# Define the U-Net model
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder_transpose = nn.ModuleList()
        self.decoder_attn = nn.ModuleList()
        self.decoder_residual = nn.ModuleList()
        self.decoder_upsample = nn.ModuleList()
        
        channels = [16, 32, 64, 80, 96]
        upsample_ratio = [16, 8, 4, 2, 1]

        # Encoder
        for channel in channels:
            self.encoder.append(nn.Sequential(
                nn.Conv2d(in_channels, channel, kernel_size=3, padding=1),
                nn.BatchNorm2d(channel),
                nn.ReLU(inplace=True),
                ResidualBlock(channel, channel),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ))
            in_channels = channel
        
        # Decoder
        for index,channel_tuple in enumerate(zip(reversed(channels), reversed(channels[:-1]))):
            channel_1, channel_2 = channel_tuple
            self.decoder_transpose.append(nn.Sequential(
                nn.ConvTranspose2d(channel_1, channel_1, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(channel_1),
                nn.ReLU(inplace=True),
                
            ))

            self.decoder_attn.append(
                AttentionBlock(channel_1+channel_2)
                
            )
            
            self.decoder_residual.append(nn.Sequential(

                
                ResidualBlock(channel_1+channel_2, channel_2)
                
            ))

            self.decoder_upsample.append(nn.Sequential(

                nn.Conv2d(channel_2, out_channels, kernel_size=1, padding=0),
                nn.Upsample(scale_factor=  upsample_ratio[index], mode='bilinear', align_corners=True)
                
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
        for encoder_block in self.encoder:
            x = encoder_block(x)
            skip_connections.append(x)

        
        
        # Decoder
        for index,block in enumerate(zip(self.decoder_transpose, self.decoder_attn, self.decoder_residual, self.decoder_upsample)):
            transpose_block, attn_block, res_block, upsample_block = block

            if index == 0: 

                #_ = skip_connections.pop() 
                
                x = transpose_block(x)

                
                skip = skip_connections[len(skip_connections)-2 - index]
                x = torch.cat((x, skip), dim=1)
                x = attn_block(x)
                x = res_block(x)
                decoder_maps.append(upsample_block(x))

                


            else:

                skip = skip_connections[len(skip_connections)-2 - index]
                x = transpose_block(x)
                x = torch.cat((x, skip), dim=1)

                x = attn_block(x)
                x = res_block(x)

                decoder_maps.append(upsample_block(x))

        
        # Final convolution
        x = self.final_conv(x)
        
        return x, decoder_maps, skip_connections
    


# Define the U-Net model
class DoubleUNet(nn.Module):

    def __init__(self, in_channels, out_channels, concat_features_encoder = False, encoder_attention= False , model_1 = None):


        super(DoubleUNet, self).__init__()
        self.model_1 = model_1
        self.in_channels = in_channels

        #initialise model_1 if not provided
        if model_1 is None:
            self.model_1 = UNet(in_channels-1, out_channels)
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
        
        channels = [16, 32, 64, 80, 96]
        upsample_ratio = [16, 8, 4, 2, 1]

        # Encoder

        if self.encoder_attention == 1:
            for index, channel in enumerate(channels):
                self.encoder_attn.append(AttentionBlock_3(channel))

        else: 
            for index, channel in enumerate(channels):
                self.encoder_attn.append(AttentionBlock_3(channel*2,concat=True))


        
        if concat_features_encoder:

            channels_enc = [in_channels,16, 32, 64, 80, 96]

            for index,channel in enumerate(channels_enc[:-1]):
            
                if index == 0:

                    self.encoder.append(nn.Sequential(
                        nn.Conv2d(channels_enc[index], channels_enc[index+1], kernel_size=3, padding=1),
                        nn.BatchNorm2d(channels_enc[index+1]),
                        nn.ReLU(inplace=True),
                        ResidualBlock(channels_enc[index+1], channels_enc[index+1]),
                        nn.MaxPool2d(kernel_size=2, stride=2)
                    ))
                    
                else:
                        
                    self.encoder.append(nn.Sequential(
                        nn.Conv2d(channels_enc[index]*2, channels_enc[index+1], kernel_size=3, padding=1),
                        nn.BatchNorm2d(channels_enc[index+1]),
                        nn.ReLU(inplace=True),
                        ResidualBlock(channels_enc[index+1], channels_enc[index+1]),
                        nn.MaxPool2d(kernel_size=2, stride=2)
                    ))


        else:

            for index,channel in enumerate(channels):

                self.encoder.append(nn.Sequential(
                    nn.Conv2d(in_channels, channel, kernel_size=3, padding=1),
                    nn.BatchNorm2d(channel),
                    nn.ReLU(inplace=True),
                    ResidualBlock(channel, channel),
                    nn.MaxPool2d(kernel_size=2, stride=2)
                ))
                in_channels = channel
       
        # Decoder
        for index,channel_tuple in enumerate(zip(reversed(channels), reversed(channels[:-1]))):
            channel_1, channel_2 = channel_tuple


            self.decoder_transpose.append(nn.Sequential(
                nn.ConvTranspose2d(channel_1, channel_1, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(channel_1),
                nn.ReLU(inplace=True),
                
            ))

            self.decoder_attn.append(nn.Sequential(
                AttentionBlock(channel_1+channel_2)
            ))

        
        
            self.decoder_residual.append(nn.Sequential(

                
                ResidualBlock(channel_1+channel_2, channel_2)
                
            ))

            self.decoder_upsample.append(nn.Sequential(

                nn.Conv2d(channel_2, out_channels, kernel_size=1, padding=0),
                nn.Upsample(scale_factor=  upsample_ratio[index], mode='bilinear', align_corners=True)
                
            ))

        
    
            # Final convolutional layer
            self.final_conv = nn.Sequential(
                                nn.Conv2d(channels[0], out_channels, kernel_size=1),
                                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                                )

    def forward(self, x):
        skip_connections = []
        decoder_maps = []

        x_1, decoder_maps_1, skip_connections_1 = self.model_1(x[:,:,:,:])
        seg_map = nn.Sigmoid()(x)

        

        
        # Encoder
        for index,encoder_block in enumerate(self.encoder):
            
            if index == 0:

                
                x = torch.cat((x, seg_map), dim=1)
                x = encoder_block(x)


            else:

                if self.concat_features_encoder:
                    
                    x = torch.cat((x, skip_connections_1[index-1]), dim=1)

                if self.encoder_attention:


                    x = self.encoder_attn[index-1](x,skip_connections_1[index-1])
                x = encoder_block(x)

                
                    

            skip_connections.append(x)
        
        # Decoder
        for index,block in enumerate(zip(self.decoder_transpose, self.decoder_attn, self.decoder_residual, self.decoder_upsample)):
            transpose_block, attn_block, res_block, upsample_block = block

            if index == 0: 

                _ = skip_connections.pop() #skip and x are the same feature map
                
                x = transpose_block(x)


                skip = skip_connections.pop()
                x = torch.cat((x, skip), dim=1)
                x = attn_block(x)
                x = res_block(x)
                decoder_maps.append(upsample_block(x))

                


            else:

                skip = skip_connections.pop()
                x = transpose_block(x)
                x = torch.cat((x, skip), dim=1)

                x = attn_block(x)
                x = res_block(x)

                decoder_maps.append(upsample_block(x))

        
        # Final convolution
        x = self.final_conv(x)

        decoder_maps = decoder_maps + decoder_maps_1 + [x_1]
        
        return (x,x_1), decoder_maps

# Example usage
in_channels = 2
out_channels = 1  # Change to the desired number of output channels
model = UNet(in_channels, out_channels)

# dummy input
x = torch.randn(1, 2, 256, 256)
output,decoder_maps,_ = model(x)


