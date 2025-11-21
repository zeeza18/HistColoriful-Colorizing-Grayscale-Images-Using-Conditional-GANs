# ============================================================================
# models.py - Model Architecture Definitions
# ============================================================================

import torch
import torch.nn as nn


# ============================================================================
# GENERATOR (U-Net Architecture)
# ============================================================================


class UNetDown(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Generator(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, n_down=8):
        super(Generator, self).__init__()
        
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)
        
        self.up_bottleneck = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.up6 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, 2, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        
        u_bottleneck = self.up_bottleneck(d8)
        u1 = self.up1(torch.cat([u_bottleneck, d7], dim=1))
        u2 = self.up2(torch.cat([u1, d6], dim=1))
        u3 = self.up3(torch.cat([u2, d5], dim=1))
        u4 = self.up4(torch.cat([u3, d4], dim=1))
        u5 = self.up5(torch.cat([u4, d3], dim=1))
        u6 = self.up6(torch.cat([u5, d2], dim=1))
        
        return self.final(torch.cat([u6, d1], dim=1))


# ============================================================================
# DISCRIMINATOR (PatchGAN)
# ============================================================================


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, n_down=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.Conv2d(512, 1, kernel_size=4, padding=1)
        )

    def forward(self, x):
        return self.model(x)


# ============================================================================
# MAIN MODEL (Combines Generator + Discriminator + Loss)
# ============================================================================


class MainModel(nn.Module):
    def __init__(self, generator=None, gen_lr=2e-4, disc_lr=2e-4, lambda_l1=100):
        super(MainModel, self).__init__()
        self.lambda_l1 = lambda_l1
        self.generator = generator or Generator(in_channels=1, out_channels=2, n_down=8)
        self.discriminator = Discriminator(in_channels=3, n_down=3)
        self.gan_loss = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()
        self.gen_optimizer = torch.optim.Adam(self.generator.parameters(), lr=gen_lr, betas=(0.5, 0.999))
        self.disc_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=disc_lr, betas=(0.5, 0.999))

    def forward(self, L):
        return self.generator(L)

    def setup_input(self, data):
        self.L = data['L']
        self.ab = data['ab']

    def forward_pass(self):
        self.fake_ab = self.generator(self.L)

    def backward_discriminator(self):
        fake_image = torch.cat([self.L, self.fake_ab], dim=1).detach()
        fake_pred = self.discriminator(fake_image)
        real_image = torch.cat([self.L, self.ab], dim=1)
        real_pred = self.discriminator(real_image)
        batch_size = self.L.size(0)
        real_labels = torch.ones(real_pred.size()).to(self.L.device)
        fake_labels = torch.zeros(fake_pred.size()).to(self.L.device)
        disc_loss_real = self.gan_loss(real_pred, real_labels)
        disc_loss_fake = self.gan_loss(fake_pred, fake_labels)
        self.disc_loss = (disc_loss_real + disc_loss_fake) * 0.5
        self.disc_loss.backward()
        return self.disc_loss.item(), disc_loss_real.item(), disc_loss_fake.item()

    def backward_generator(self):
        fake_image = torch.cat([self.L, self.fake_ab], dim=1)
        fake_pred = self.discriminator(fake_image)
        real_labels = torch.ones(fake_pred.size()).to(self.L.device)
        loss_G_GAN = self.gan_loss(fake_pred, real_labels)
        loss_G_L1 = self.l1_loss(self.fake_ab, self.ab) * self.lambda_l1
        self.loss_G = loss_G_GAN + loss_G_L1
        self.loss_G.backward()
        return self.loss_G.item(), loss_G_GAN.item(), loss_G_L1.item()

    def optimize_parameters(self):
        self.forward_pass()
        self.disc_optimizer.zero_grad()
        disc_loss, disc_real, disc_fake = self.backward_discriminator()
        self.disc_optimizer.step()
        self.gen_optimizer.zero_grad()
        gen_loss, gen_gan, gen_l1 = self.backward_generator()
        self.gen_optimizer.step()
        return {
            'disc_loss': disc_loss,
            'disc_loss_real': disc_real,
            'disc_loss_gen': disc_fake,
            'loss_G': gen_loss,
            'loss_G_GAN': gen_gan,
            'loss_G_L1': gen_l1
        }


# ============================================================================
# HELPER FUNCTION TO LOAD MODEL
# ============================================================================


def map_checkpoint_keys(checkpoint):
    new_state_dict = {}
    
    for old_key, value in checkpoint.items():
        if not old_key.startswith('generator.'):
            continue
            
        key = old_key.replace('generator.model.model.', '')
        
        if key.startswith('0.'):
            new_key = 'down1.model.' + key
        elif key.startswith('1.model.1.'):
            new_key = 'down2.model.0.' + key.split('1.model.1.')[1]
        elif key.startswith('1.model.2.'):
            new_key = 'down2.model.1.' + key.split('1.model.2.')[1]
        elif key.startswith('1.model.3.model.1.'):
            new_key = 'down3.model.0.' + key.split('1.model.3.model.1.')[1]
        elif key.startswith('1.model.3.model.2.'):
            new_key = 'down3.model.1.' + key.split('1.model.3.model.2.')[1]
        elif key.startswith('1.model.3.model.3.model.1.'):
            new_key = 'down4.model.0.' + key.split('1.model.3.model.3.model.1.')[1]
        elif key.startswith('1.model.3.model.3.model.2.'):
            new_key = 'down4.model.1.' + key.split('1.model.3.model.3.model.2.')[1]
        elif key.startswith('1.model.3.model.3.model.3.model.1.'):
            new_key = 'down5.model.0.' + key.split('1.model.3.model.3.model.3.model.1.')[1]
        elif key.startswith('1.model.3.model.3.model.3.model.2.'):
            new_key = 'down5.model.1.' + key.split('1.model.3.model.3.model.3.model.2.')[1]
        elif key.startswith('1.model.3.model.3.model.3.model.3.model.1.'):
            new_key = 'down6.model.0.' + key.split('1.model.3.model.3.model.3.model.3.model.1.')[1]
        elif key.startswith('1.model.3.model.3.model.3.model.3.model.2.'):
            new_key = 'down6.model.1.' + key.split('1.model.3.model.3.model.3.model.3.model.2.')[1]
        elif key.startswith('1.model.3.model.3.model.3.model.3.model.3.model.1.'):
            new_key = 'down7.model.0.' + key.split('1.model.3.model.3.model.3.model.3.model.3.model.1.')[1]
        elif key.startswith('1.model.3.model.3.model.3.model.3.model.3.model.2.'):
            new_key = 'down7.model.1.' + key.split('1.model.3.model.3.model.3.model.3.model.3.model.2.')[1]
        elif key.startswith('1.model.3.model.3.model.3.model.3.model.3.model.3.model.1.'):
            new_key = 'down8.model.0.' + key.split('1.model.3.model.3.model.3.model.3.model.3.model.3.model.1.')[1]
        elif key.startswith('1.model.3.model.3.model.3.model.3.model.3.model.3.model.3.'):
            new_key = 'up_bottleneck.0.' + key.split('1.model.3.model.3.model.3.model.3.model.3.model.3.model.3.')[1]
        elif key.startswith('1.model.3.model.3.model.3.model.3.model.3.model.3.model.4.'):
            new_key = 'up_bottleneck.1.' + key.split('1.model.3.model.3.model.3.model.3.model.3.model.3.model.4.')[1]
        elif key.startswith('1.model.3.model.3.model.3.model.3.model.3.model.5.'):
            new_key = 'up1.0.' + key.split('1.model.3.model.3.model.3.model.3.model.3.model.5.')[1]
        elif key.startswith('1.model.3.model.3.model.3.model.3.model.3.model.6.'):
            new_key = 'up1.1.' + key.split('1.model.3.model.3.model.3.model.3.model.3.model.6.')[1]
        elif key.startswith('1.model.3.model.3.model.3.model.3.model.5.'):
            new_key = 'up2.0.' + key.split('1.model.3.model.3.model.3.model.3.model.5.')[1]
        elif key.startswith('1.model.3.model.3.model.3.model.3.model.6.'):
            new_key = 'up2.1.' + key.split('1.model.3.model.3.model.3.model.3.model.6.')[1]
        elif key.startswith('1.model.3.model.3.model.3.model.5.'):
            new_key = 'up3.0.' + key.split('1.model.3.model.3.model.3.model.5.')[1]
        elif key.startswith('1.model.3.model.3.model.3.model.6.'):
            new_key = 'up3.1.' + key.split('1.model.3.model.3.model.3.model.6.')[1]
        elif key.startswith('1.model.3.model.3.model.5.'):
            new_key = 'up4.0.' + key.split('1.model.3.model.3.model.5.')[1]
        elif key.startswith('1.model.3.model.3.model.6.'):
            new_key = 'up4.1.' + key.split('1.model.3.model.3.model.6.')[1]
        elif key.startswith('1.model.3.model.5.'):
            new_key = 'up5.0.' + key.split('1.model.3.model.5.')[1]
        elif key.startswith('1.model.3.model.6.'):
            new_key = 'up5.1.' + key.split('1.model.3.model.6.')[1]
        elif key.startswith('1.model.5.'):
            new_key = 'up6.0.' + key.split('1.model.5.')[1]
        elif key.startswith('1.model.6.'):
            new_key = 'up6.1.' + key.split('1.model.6.')[1]
        elif key.startswith('3.'):
            new_key = 'final.0.' + key.split('3.')[1]
        else:
            print(f"Warning: Unmatched key: {old_key}")
            continue
        
        new_state_dict[new_key] = value
    
    return new_state_dict


def load_colorization_model(model_path, device='cpu'):
    try:
        print(f"Loading checkpoint from: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        print(f"Mapping checkpoint keys...")
        generator_state_dict = map_checkpoint_keys(checkpoint)
        
        print(f"Mapped {len(generator_state_dict)} parameters")
        
        generator = Generator(in_channels=1, out_channels=2, n_down=8)
        
        missing_keys, unexpected_keys = generator.load_state_dict(generator_state_dict, strict=False)
        
        if missing_keys:
            print(f"⚠️  Missing keys: {len(missing_keys)}")
            for key in missing_keys[:5]:
                print(f"   - {key}")
        
        if unexpected_keys:
            print(f"⚠️  Unexpected keys: {len(unexpected_keys)}")
            for key in unexpected_keys[:5]:
                print(f"   - {key}")
        
        generator.eval()
        generator.to(device)
        
        model = MainModel(
            generator=generator,
            gen_lr=2e-4,
            disc_lr=2e-4,
            lambda_l1=50
        )
        model.to(device)
        
        print(f"✓ Model loaded successfully")
        print(f"✓ Device: {device}")
        print(f"✓ Generator parameters: {sum(p.numel() for p in model.generator.parameters()):,}")
        
        return model
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print(f"Initializing with random weights as fallback...")
        
        generator = Generator(in_channels=1, out_channels=2, n_down=8)
        generator.eval()
        generator.to(device)
        
        model = MainModel(
            generator=generator,
            gen_lr=2e-4,
            disc_lr=2e-4,
            lambda_l1=50
        )
        model.to(device)
        
        print(f"✓ Model initialized with random weights")
        print(f"✓ Device: {device}")
        return model