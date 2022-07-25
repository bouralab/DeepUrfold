import torch
import torch.nn as nn
import sparseconvnet as scn
import torch.nn.functional as F

class AutoencoderModel(nn.Module):
    """
    dimension = 3
    reps = 1 #Conv block repetition factor
    m = 32 #Unet number of features
    nPlanes = [m, 2*m, 3*m, 4*m, 5*m] #UNet number of features per level
    """
    def __init__(self, nIn,  sz=256, dimension=3, reps=1, m=32, nLevels=5, nPlanes=None, skipconnection=True):
        nn.Module.__init__(self)
        if nPlanes is None:
            nPlanes = [l*m for l in range(1,nLevels+1)]
        self.sparseModel = scn.Sequential().add(
           scn.InputLayer(dimension, torch.LongTensor([sz]*3), mode=3)).add(
           scn.SubmanifoldConvolution(dimension, nIn, m, 3, False)).add(
           UNet(dimension, reps, nPlanes,  residual_blocks=True, downsample=[2,2], skip_connections=skipconnection)).add(
           scn.BatchNormReLU(m)).add(
           scn.OutputLayer(dimension))
        self.linear = nn.Linear(m, nIn)
    def forward(self,x):
        x=self.sparseModel(x)
        x=self.linear(x)
        return x

class SegmenterAutoEncoderModel(nn.Module):
    """
    dimension = 3
    reps = 1 #Conv block repetition factor
    m = 32 #Unet number of features
    nPlanes = [m, 2*m, 3*m, 4*m, 5*m] #UNet number of features per level
    """
    def __init__(self, nIn, nOut=2, sz=256, dimension=3, reps=1, m=32, nLevels=5, nPlanes=None):
        nn.Module.__init__(self)
        if nPlanes is None:
            nPlanes = [l*m for l in range(1,nLevels+1)]

        self.sparseModel = scn.Sequential().add(
           scn.InputLayer(dimension, torch.LongTensor([sz]*3), mode=3)).add(
           scn.SubmanifoldConvolution(dimension, nIn, m, 3, False)).add(
           UNet(dimension, reps, nPlanes, residual_blocks=True, downsample=[2,2])).add(
           scn.BatchNormReLU(m)).add(
           scn.OutputLayer(dimension))
        self.linear_segmenter = nn.Linear(m, nOut)
        self.activation_segmenter = nn.Softmax(dim=1)
        self.linear_reconstructer = nn.Linear(m, nIn)
        #self.activation_reconstructer = nn.Softmax(dim=1)
    def forward(self,x):
        x=self.sparseModel(x)
        seg=self.linear_segmenter(x)
        seg=self.activation_segmenter(seg)
        rec=self.linear_reconstructer(x)
        rec=self.activation_reconstructer(seg)
        return seg, rec

class CVAE(nn.Module):
    ''' This the VAE, which takes a encoder and decoder.

    '''
    def __init__(self, nIn, sz=256, dimension=3, reps=1, m=32, nLevels=5,
      nPlanes=None, nLatent=2, dropout=0.5):
        super().__init__()
        if nPlanes is None:
            nPlanes = [l*m for l in range(1,nLevels+1)]
        self.nLatent = nLatent

        self.input = scn.Sequential() \
           .add(scn.InputLayer(dimension, torch.LongTensor([sz]*3), mode=3)) \
           .add(scn.SubmanifoldConvolution(dimension, nIn, m, 3, False))

        self.encoder, self.decoder = VariationalUNetAutoEncoder(dimension, reps,
            nPlanes, residual_blocks=True, downsample=[2, 2], dropout=dropout,
            nLatent=nLatent)

        self.output = scn.Sequential() \
            .add(scn.BatchNormReLU(m)) \
            .add(scn.OutputLayer(dimension)) \
            .add(nn.Linear(m, nIn)) \

    @staticmethod
    def ELBO(x, reconstructed_x, mean, log_var):
        # reconstruction loss
        RCL = F.binary_cross_entropy(reconstructed_x, x, size_average=False)
        # kl divergence loss
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        return RCL + KLD

    def forward(self, x):

        x = self.input(x)

        # encode
        z_mu, z_var = self.encoder(x)

        # sample from the distribution having latent parameters z_mu, z_var
        # reparameterize
        std = torch.exp(z_var / 2)
        eps = torch.randn_like(std)
        x_sample = eps.mul(std).add_(z_mu)

        z = x_sample

        # decode
        generated_x = self.decoder(z)
        generated_x = self.output(generated_x)

        return generated_x, z_mu, z_var

    def sample(gpu=False):
        device = torch.device('cuda' if torch.cuda.is_available() and gpu else 'cpu')

        # create a random latent vector
        z = torch.randn(1, self.nLatent).to(device)

        reconstructed_x = self.decoder(z)
        reconstructed_x = self.output(reconstructed_x)

        return reconstructed_x

class VariationalBayesEnocoder(nn.Module):
    def __init__(self, nHidden, nLatent):
        super().__init__()
        self.mu = nn.Linear(nHidden, nLatent)
        self.var = nn.Linear(nHidden, nLatent)

    def forward(self, x):
        # x is of hidden shape

        # latent parameters
        mean = self.mu(x)
        # mean is of shape [batch_size, latent_dim]
        log_var = self.var(x)
        # log_var is of shape [batch_size, latent_dim]

        return mean, log_var

class VariationalBayesDecoder(nn.Module):
    def __init__(self, nLatent, nHidden):
        '''
        Args:
            latent_dim: A integer indicating the latent size.
            hidden_dim: A integer indicating the size of hidden dimension.
            output_dim: A integer indicating the size of output (in case of MNIST 28 * 28).
            n_classes: A integer indicating the number of classes. (dimension of one-hot representation of labels)
        '''
        super().__init__()
        self.latent_to_hidden = nn.Linear(nLatent, nHidden)

    def forward(self, x):
        x = F.relu(self.latent_to_hidden(x))

        return x

class KNNLayer(nn.Module):
    def __init__(self, dimension, spatial_size, mode=3):
        Module.__init__(self)
        self.mode = mode
        self.device = None

        # resource object, can be re-used over calls
        self.res = faiss.StandardGpuResources()
        # put on same stream as pytorch to avoid synchronizing streams
        self.res.setDefaultNullStreamAllDevices()

    def to(self, device):
        self.device=device
        return self

    def forward(self, input):
        output = SparseConvNetTensor(
            metadata=Metadata(
                self.dimension),
            spatial_size=self.spatial_size)
        output.features = InputLayerFunction.apply(
            self.dimension,
            output.metadata,
            self.spatial_size,
            input[0].cpu().long(),
            input[1].to(self.device) if self.device else input[1],
            0 if len(input) == 2 else input[2],
            self.mode
        )
        return output

def UNet(dimension, reps, nPlanes, residual_blocks=False, downsample=[2, 2],
  leakiness=0, n_input_planes=-1, dropout=0.5, skip_connections=True):
    """
    U-Net style network with VGG or ResNet-style blocks.
    For voxel level prediction:
    import sparseconvnet as scn
    import torch.nn
    class Model(nn.Module):
        def __init__(self):
            nn.Module.__init__(self)
            self.sparseModel = scn.Sequential().add(
               scn.SubmanifoldConvolution(3, nInputFeatures, 64, 3, False)).add(
               scn.UNet(3, 2, [64, 128, 192, 256], residual_blocks=True, downsample=[2, 2]))
            self.linear = nn.Linear(64, nClasses)
        def forward(self,x):
            x=self.sparseModel(x).features
            x=self.linear(x)
            return x
    """
    def block(m, a, b):
        if residual_blocks: #ResNet style blocks
            m.add(scn.ConcatTable()
                  .add(scn.Identity() if a == b else scn.NetworkInNetwork(a, b, False))
                  .add(scn.Sequential()
                    .add(scn.BatchNormLeakyReLU(a,leakiness=leakiness))
                    .add(scn.Dropout(dropout))
                    .add(scn.SubmanifoldConvolution(dimension, a, b, 3, False))
                    .add(scn.BatchNormLeakyReLU(b,leakiness=leakiness))
                    .add(scn.Dropout(dropout))
                    .add(scn.SubmanifoldConvolution(dimension, b, b, 3, False)))
             ).add(scn.AddTable())
        else: #VGG style blocks
            m.add(scn.Sequential()
                 .add(scn.BatchNormLeakyReLU(a,leakiness=leakiness))
                 .add(scn.Dropout(dropout))
                 .add(scn.SubmanifoldConvolution(dimension, a, b, 3, False)))
    def U(nPlanes,n_input_planes=-1): #Recursive function
        m = scn.Sequential()
        for i in range(reps):
            block(m, n_input_planes if n_input_planes!=-1 else nPlanes[0], nPlanes[0])
            n_input_planes=-1
        if len(nPlanes) > 1:
            concater = scn.ConcatTable() if skip_connections else scn.Sequential()
            m.add(
                concater.add(
                    scn.Identity()).add(
                    scn.Sequential().add(
                        scn.BatchNormLeakyReLU(nPlanes[0],leakiness=leakiness)).add(
                        scn.Convolution(dimension, nPlanes[0], nPlanes[1],
                            downsample[0], downsample[1], False)).add(
                        scn.Dropout(dropout)).add(
                        U(nPlanes[1:])).add(
                        scn.BatchNormLeakyReLU(nPlanes[1],leakiness=leakiness)).add(
                        scn.Deconvolution(dimension, nPlanes[1], nPlanes[0],
                            downsample[0], downsample[1], False))))
            if skip_connections:
                m.add(scn.JoinTable())
            for i in range(reps):
                block(m, nPlanes[0] * (2 if i == 0 and skip_connections else 1), nPlanes[0])
        return m
    m = U(nPlanes,n_input_planes)
    return m

def VariationalUNetAutoEncoder(dimension, reps, nPlanes, residual_blocks=False,
  downsample=[2, 2], leakiness=0, n_input_planes=-1, dropout=0.5, nLatent=2):
    """
    U-Net style network with VGG or ResNet-style blocks.
    For voxel level prediction:
    import sparseconvnet as scn
    import torch.nn
    class Model(nn.Module):
        def __init__(self):
            nn.Module.__init__(self)
            self.sparseModel = scn.Sequential().add(
               scn.SubmanifoldConvolution(3, nInputFeatures, 64, 3, False)).add(
               scn.UNet(3, 2, [64, 128, 192, 256], residual_blocks=True, downsample=[2, 2]))
            self.linear = nn.Linear(64, nClasses)
        def forward(self,x):
            x=self.sparseModel(x).features
            x=self.linear(x)
            return x
    """
    def block(m, a, b):
        if residual_blocks: #ResNet style blocks
            m.add(scn.ConcatTable()
                  .add(scn.Identity() if a == b else scn.NetworkInNetwork(a, b, False))
                  .add(scn.Sequential()
                    .add(scn.BatchNormLeakyReLU(a,leakiness=leakiness))
                    .add(scn.Dropout(dropout))
                    .add(scn.SubmanifoldConvolution(dimension, a, b, 3, False))
                    .add(scn.BatchNormLeakyReLU(b,leakiness=leakiness))
                    .add(scn.Dropout(dropout))
                    .add(scn.SubmanifoldConvolution(dimension, b, b, 3, False)))
             ).add(scn.AddTable())
        else: #VGG style blocks
            m.add(scn.Sequential()
                 .add(scn.BatchNormLeakyReLU(a,leakiness=leakiness))
                 .add(scn.Dropout(dropout))
                 .add(scn.SubmanifoldConvolution(dimension, a, b, 3, False)))
    def U(nPlanes,n_input_planes=-1): #Recursive function
        encoder = scn.Sequential()
        decoder = scn.Sequential()
        #if i > 0:
        for i in range(reps):
            block(encoder, n_input_planes if n_input_planes!=-1 else nPlanes[0], nPlanes[0])
            n_input_planes=-1
        if len(nPlanes) > 1:
            next_encoder, next_decoder = U(nPlanes[1:])
            encoder.add(
                scn.BatchNormLeakyReLU(nPlanes[0],leakiness=leakiness).add(
                scn.Convolution(dimension, nPlanes[0], nPlanes[1],
                    downsample[0], downsample[1], False)).add(
                scn.Dropout(dropout)).add(
                next_encoder))
            decoder.add(
                next_decoder.add(
                scn.BatchNormLeakyReLU(nPlanes[1],leakiness=leakiness)).add(
                scn.Deconvolution(dimension, nPlanes[1], nPlanes[0],
                    downsample[0], downsample[1], False)))
            decoder.add(scn.JoinTable())
            for i in range(reps):
                block(decoder, nPlanes[0] * (2 if i == 0 else 1), nPlanes[0])
        elif len(nPlanes) == 1:
            #At bottleneck
            encoder.add(VariationalBayesEnocoder(nPlanes[0], nLatent))
            decoder.add(VariationalBayesDecoder(nLatent, nPlanes[0]))
        return encoder, decoder
    encoder, decoder = U(nPlanes,n_input_planes)
    return encoder, decoder