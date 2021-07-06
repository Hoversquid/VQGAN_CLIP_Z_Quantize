import argparse
import math
from pathlib import Path
import sys
from io import BytesIO
import requests
sys.path.append('./taming-transformers')

from IPython import display
from omegaconf import OmegaConf
from PIL import Image
from taming.models import cond_transformer, vqgan
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm.notebook import tqdm
from os import chdir, mkdir, path, getcwd, walk
from os.path import isfile
from CLIP import clip

class VQGAN_CLIP_Z_Quantize:
    def __init__(self, Other_txt_prompts,
                Other_img_prompts,
                Other_noise_seeds,
                Other_noise_weights,
                Output_directory,
                Base_Image, Base_Image_Weight,
                Image_Prompt1, Image_Prompt2, Image_Prompt3,
                Text_Prompt1,Text_Prompt2,Text_Prompt3,
                SizeX, SizeY,
                Noise_Seed_Number, Noise_Weight, Display_Frequency):
        try:
          Noise_Seed_Number = int(Noise_Seed_Number)
          noise_prompt_seeds = [Noise_Seed_Number] + Other_noise_seeds
          noise_prompt_weights = [Noise_Weight] + Other_noise_weights
        except:
          print("No noise seeds used.")
          noise_prompt_seeds = Other_noise_seeds
          noise_prompt_weights = Other_noise_weights

        txt_prompts = self.get_prompt_list(Text_Prompt1, Text_Prompt2, Text_Prompt3, Other_txt_prompts)
        img_prompts = self.get_prompt_list(Image_Prompt1, Image_Prompt2, Image_Prompt3, Other_img_prompts)

        self.args = argparse.Namespace(
            outdir=Output_directory, # this is then name of where your output will go in /content or in /MyDrive
            init_image=Base_Image,
            init_weight=Base_Image_Weight,
            prompts=txt_prompts,
            image_prompts=img_prompts,
            noise_prompt_seeds=noise_prompt_seeds,
            noise_prompt_weights=noise_prompt_weights,
            size=[SizeX, SizeY],
            clip_model='ViT-B/32',
            vqgan_config='vqgan_imagenet_f16_1024.yaml',
            vqgan_checkpoint='vqgan_imagenet_f16_1024.ckpt',
            step_size=0.05,
            cutn=64,
            cut_pow=1.,
            display_freq=Display_Frequency,
            seed=0
        )

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('Using device:', device)

        model = self.load_vqgan_model(self.args.vqgan_config, self.args.vqgan_checkpoint).to(device)
        perceptor = clip.load(self.args.clip_model, jit=False)[0].eval().requires_grad_(False).to(device)

        cut_size = perceptor.visual.input_resolution
        e_dim = model.quantize.e_dim
        f = 2**(model.decoder.num_resolutions - 1)
        make_cutouts = MakeCutouts(self, cut_size, self.args.cutn, cut_pow=self.args.cut_pow)
        n_toks = model.quantize.n_e
        toksX, toksY = self.args.size[0] // f, self.args.size[1] // f
        sideX, sideY = toksX * f, toksY * f
        z_min = model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
        z_max = model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]

        if self.args.seed is not None:
            torch.manual_seed(self.args.seed)

        imgpath = None
        if not self.args.init_image in (None, ""):
          imgpath = self.get_pil_imagepath(self.args.init_image)

        if imgpath:
            pil_image = Image.open(imgpath).convert('RGB')
            pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
            z, *_ = model.encode(TF.to_tensor(pil_image).to(device).unsqueeze(0) * 2 - 1)
        else:
            one_hot = F.one_hot(torch.randint(n_toks, [toksY * toksX], device=device), n_toks).float()
            z = one_hot @ model.quantize.embedding.weight
            z = z.view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2)
        z_orig = z.clone()
        z.requires_grad_(True)
        opt = optim.Adam([z], lr=self.args.step_size)

        normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                         std=[0.26862954, 0.26130258, 0.27577711])
        pMs = []

        filename = ""
        name_limit = 42
        for i, prompt in enumerate(self.args.prompts):
            name_length = name_limit - len(filename)
            if name_length > 0:
              filename += prompt[:name_length]
              if len(filename) + 2 < name_limit and i + 1 < len(self.args.prompts):
                filename += "__"


            txt, weight, stop = self.parse_prompt(prompt)
            embed = perceptor.encode_text(clip.tokenize(txt).to(device)).float()
            pMs.append(Prompt(embed, weight, stop).to(device))

        if filename == "":
          filename = "No_Prompts"

        for prompt in self.args.image_prompts:
            imgpath, weight, stop = self.parse_prompt(prompt)
            imgpath = self.get_pil_imagepath(imgpath)

            img = self.resize_image(Image.open(imgpath).convert('RGB'), (sideX, sideY))
            batch = make_cutouts(TF.to_tensor(img).unsqueeze(0).to(device))
            embed = perceptor.encode_image(normalize(batch)).float()
            pMs.append(Prompt(embed, weight, stop).to(device))

        for seed, weight in zip(self.args.noise_prompt_seeds, self.args.noise_prompt_weights):
            gen = torch.Generator().manual_seed(seed)
            embed = torch.empty([1, perceptor.visual.output_dim]).normal_(generator=gen)
            pMs.append(Prompt(embed, weight).to(device))

        i = 0

        filename = filename.replace(" ", "_")
        if not path.exists(self.args.outdir):
          mkdir(self.args.outdir)
        outname = self.set_valid_filename(self.args.outdir, filename, 0)

        filelistpath = path.join(self.args.outdir, outname + ".txt")

        self.write_arg_list(self.args)
        try:
          with tqdm() as pbar:
            while True:
                self.train(i, outname)
                i += 1
                pbar.update()
        except KeyboardInterrupt:
            pass

    def load_vqgan_model(self, config_path, checkpoint_path):
        config = OmegaConf.load(config_path)
        if config.model.target == 'taming.models.vqgan.VQModel':
            model = vqgan.VQModel(**config.model.params)
            model.eval().requires_grad_(False)
            model.init_from_ckpt(checkpoint_path)
        elif config.model.target == 'taming.models.cond_transformer.Net2NetTransformer':
            parent_model = cond_transformer.Net2NetTransformer(**config.model.params)
            parent_model.eval().requires_grad_(False)
            parent_model.init_from_ckpt(checkpoint_path)
            model = parent_model.first_stage_model
        else:
            raise ValueError(f'unknown model type: {config.model.target}')
        del model.loss
        return model

    def resize_image(self, image, out_size):
        ratio = image.size[0] / image.size[1]
        area = min(image.size[0] * image.size[1], out_size[0] * out_size[1])
        size = round((area * ratio)**0.5), round((area / ratio)**0.5)
        return image.resize(size, Image.LANCZOS)

    def synth(z):
        z_q = vector_quantize(z.movedim(1, 3), model.quantize.embedding.weight).movedim(3, 1)
        return ClampWithGrad.apply(model.decode(z_q).add(1).div(2), 0, 1)

    @torch.no_grad()
    def checkin(self, i, losses, name):
        losses_str = ', '.join(f'{loss.item():g}' for loss in losses)
        tqdm.write(f'i: {i}, loss: {sum(losses).item():g}, losses: {losses_str}')
        out = synth(z)
        sequence_number = i // self.args.display_freq
        outname = path.join(self.args.outdir, name)
        outname = image_output_path(outname, sequence_number=sequence_number)

        TF.to_pil_image(out[0].cpu()).save(outname)
        display.display(display.Image(str(outname)))

    def ascend_txt(self):
        out = synth(z)
        iii = perceptor.encode_image(normalize(make_cutouts(out))).float()

        result = []

        if self.args.init_weight:
            result.append(F.mse_loss(z, z_orig) * self.args.init_weight / 2)

        for prompt in pMs:
            result.append(prompt(iii))

        return result

    def train(self, i, name):
        opt.zero_grad()
        lossAll = ascend_txt()
        if i % self.args.display_freq == 0:
            checkin(i, lossAll, name)
        loss = sum(lossAll)
        loss.backward()
        opt.step()
        with torch.no_grad():
            z.copy_(z.maximum(z_min).minimum(z_max))

    # Used to set image path if it's a URL
    def get_pil_imagepath(self, imgpath):
        if not path.exists(imgpath):
          imgpath = requests.get(imgpath, stream=True).raw
        return imgpath

    # Added for progress saving
    def image_output_path(self, output_path, sequence_number=None):
        """
        Returns underscore separated Path.
        A current timestamp is prepended if `self.save_date_time` is set.
        Sequence number left padded with 6 zeroes is appended if `save_every` is set.
        :rtype: Path
        """
        # output_path = self.textpath
        if sequence_number:
            sequence_number_left_padded = str(sequence_number).zfill(6)
            output_path = f"{output_path}.{sequence_number_left_padded}"
        return Path(f"{output_path}.png")

    def set_valid_filename(self, filename, basename, i):
        if i > 0:
            newname = "%s(%d)" % (basename, i)
        else:
            newname = basename

        unique_name = True
        for root, dir, files in walk(self.args.outdir):
            for f in files:
              if path.splitext(f)[0] == newname:
                unique_name = False
                break
            if not unique_name: break

        if unique_name:
          return path.join(filename, newname)

        return self.set_valid_filename(filename, basename, i + 1)

    def get_prompt_list(self, first, second, third, rest):
      param_list = [first, second, third]
      param_list = [p for p in param_list if p]
      prompt_list = param_list + rest
      return prompt_list

    def write_arg_list(self,args):
      with open(filelistpath, "w", encoding="utf-8") as txtfile:
        argdict = vars(args)
        txt = ""
        for argname, argval in argdict.items():
          txt += f"{str(argname)}={str(argval)},\n"
        txt = f"args = argparse.Namespace(\n{txt})"
        txtfile.write(txt)

    def parse_prompt(self, prompt):
        vals = prompt.rsplit('|', 2)
        vals = vals + ['', '1', '-inf'][len(vals):]
        print(f"Vals: {vals}")
        return vals[0], float(vals[1]), float(vals[2])

    def sinc(self, x):
        return torch.where(x != 0, torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))


    def lanczos(self, x, a):
        cond = torch.logical_and(-a < x, x < a)
        out = torch.where(cond, self.sinc(x) * self.sinc(x/a), x.new_zeros([]))
        return out / out.sum()


    def ramp(self, ratio, width):
        n = math.ceil(width / ratio + 1)
        out = torch.empty([n])
        cur = 0
        for i in range(out.shape[0]):
            out[i] = cur
            cur += ratio
        return torch.cat([-out[1:].flip([0]), out])[1:-1]


    def resample(self, input, size, align_corners=True):
        n, c, h, w = input.shape
        dh, dw = size

        input = input.view([n * c, 1, h, w])

        if dh < h:
            kernel_h = self.lanczos(self.ramp(dh / h, 2), 2).to(input.device, input.dtype)
            pad_h = (kernel_h.shape[0] - 1) // 2
            input = F.pad(input, (0, 0, pad_h, pad_h), 'reflect')
            input = F.conv2d(input, kernel_h[None, None, :, None])

        if dw < w:
            kernel_w = self.lanczos(self.ramp(dw / w, 2), 2).to(input.device, input.dtype)
            pad_w = (kernel_w.shape[0] - 1) // 2
            input = F.pad(input, (pad_w, pad_w, 0, 0), 'reflect')
            input = F.conv2d(input, kernel_w[None, None, None, :])

        input = input.view([n, c, h, w])
        return F.interpolate(input, size, mode='bicubic', align_corners=align_corners)

    def vector_quantize(self, x, codebook):
        d = x.pow(2).sum(dim=-1, keepdim=True) + codebook.pow(2).sum(dim=1) - 2 * x @ codebook.T
        indices = d.argmin(-1)
        x_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
        return ReplaceGrad.apply(x_q, x)

class ReplaceGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_forward, x_backward):
        ctx.shape = x_backward.shape
        return x_forward

    @staticmethod
    def backward(ctx, grad_in):
        return None, grad_in.sum_to_size(ctx.shape)

class ClampWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min, max):
        ctx.min = min
        ctx.max = max
        ctx.save_for_backward(input)
        return input.clamp(min, max)

    @staticmethod
    def backward(ctx, grad_in):
        input, = ctx.saved_tensors
        return grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >= 0), None, None

class Prompt(nn.Module):
    def __init__(self, embed, weight=1., stop=float('-inf')):
        super().__init__()
        self.register_buffer('embed', embed)
        self.register_buffer('weight', torch.as_tensor(weight))
        self.register_buffer('stop', torch.as_tensor(stop))

    def forward(self, input):
        input_normed = F.normalize(input.unsqueeze(1), dim=2)
        embed_normed = F.normalize(self.embed.unsqueeze(0), dim=2)
        dists = input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)
        dists = dists * self.weight.sign()
        return self.weight.abs() * ReplaceGrad.apply(dists, torch.maximum(dists, self.stop)).mean()

class MakeCutouts(nn.Module):
    def __init__(self, sampler, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow
        self.sampler = sampler

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(self.sampler.resample(cutout, (self.cut_size, self.cut_size)))
        return ClampWithGrad.apply(torch.cat(cutouts, dim=0), 0, 1)
