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
from IPython.display import clear_output
from collections import OrderedDict

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
                Noise_Seed_Number, Noise_Weight, Seed,
                Display_Frequency, Clear_Interval):

        try:
          Noise_Seed_Number = int(Noise_Seed_Number)
          noise_prompt_seeds = [Noise_Seed_Number] + Other_noise_seeds
          noise_prompt_weights = [Noise_Weight] + Other_noise_weights
        except:
          print("No noise seeds used.")
          noise_prompt_seeds = Other_noise_seeds
          noise_prompt_weights = Other_noise_weights

        try:
            Seed = int(Seed)
        except:
            Seed = 0

        prompts = OrderedDict()
        prompts["Other_txt_prompts"] = Other_txt_prompts
        prompts["Other_img_prompts"] = Other_img_prompts
        prompts["Other_noise_seeds"] = Other_noise_seeds
        prompts["Other_noise_weights"] = Other_noise_weights

        arg_list = {"Output_directory":Output_directory,"Base_Image":Base_Image,
                    "Base_Image_Weight":Base_Image_Weight,
                    "Image_Prompt1":Image_Prompt1,"Image_Prompt2":Image_Prompt2,"Image_Prompt3":Image_Prompt3,
                    "Text_Prompt1":Text_Prompt1,"Text_Prompt2":Text_Prompt2,"Text_Prompt3":Text_Prompt3,
                    "SizeX":SizeX,"SizeY":SizeY,"Noise_Seed_Number":Noise_Seed_Number,
                    "Noise_Weight":Noise_Weight,"Seed":Seed,"Display_Frequency":Display_Frequency,"Clear_Interval":Clear_Interval}

        prompts.update(arg_list)

        txt_prompts = self.get_prompt_list(Text_Prompt1, Text_Prompt2, Text_Prompt3, Other_txt_prompts)
        img_prompts = self.get_prompt_list(Image_Prompt1, Image_Prompt2, Image_Prompt3, Other_img_prompts)

        self.args = argparse.Namespace(
            outdir=Output_directory, # this is the name of where your output will go
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
            seed=Seed
        )

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('Using device:', device)

        self.model = self.load_vqgan_model(self.args.vqgan_config, self.args.vqgan_checkpoint).to(device)
        self.perceptor = clip.load(self.args.clip_model, jit=False)[0].eval().requires_grad_(False).to(device)
        try:
            self.clear_interval = int(Clear_Interval)
        except:
            self.clear_interval = 0
        cut_size = self.perceptor.visual.input_resolution
        e_dim = self.model.quantize.e_dim
        f = 2**(self.model.decoder.num_resolutions - 1)
        self.make_cutouts = MakeCutouts(self, cut_size, self.args.cutn, cut_pow=self.args.cut_pow)
        n_toks = self.model.quantize.n_e
        toksX, toksY = self.args.size[0] // f, self.args.size[1] // f
        sideX, sideY = toksX * f, toksY * f
        self.z_min = self.model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
        self.z_max = self.model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]

        if self.args.seed is not None:
            torch.manual_seed(self.args.seed)

        imgpath = None
        if not self.args.init_image in (None, ""):
          imgpath = self.get_pil_imagepath(self.args.init_image)

        if imgpath:
            pil_image = Image.open(imgpath).convert('RGB')
            pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
            self.z, *_ = self.model.encode(TF.to_tensor(pil_image).to(device).unsqueeze(0) * 2 - 1)
        else:
            one_hot = F.one_hot(torch.randint(n_toks, [toksY * toksX], device=device), n_toks).float()
            self.z = one_hot @ self.model.quantize.embedding.weight
            self.z = self.z.view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2)
        self.z_orig = self.z.clone()
        self.z.requires_grad_(True)
        self.opt = optim.Adam([self.z], lr=self.args.step_size)

        self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                         std=[0.26862954, 0.26130258, 0.27577711])
        self.pMs = []

        filename = ""
        name_limit = 42
        for i, prompt in enumerate(self.args.prompts):
            name_length = name_limit - len(filename)
            if name_length > 0:
              filename += prompt[:name_length]
              if len(filename) + 2 < name_limit and i + 1 < len(self.args.prompts):
                filename += "__"


            txt, weight, stop = self.parse_prompt(prompt)
            embed = self.perceptor.encode_text(clip.tokenize(txt).to(device)).float()
            self.pMs.append(Prompt(embed, weight, stop).to(device))

        if filename == "":
          filename = "No_Prompts"

        for prompt in self.args.image_prompts:
            imgpath, weight, stop = self.parse_prompt(prompt)
            imgpath = self.get_pil_imagepath(imgpath)

            img = self.resize_image(Image.open(imgpath).convert('RGB'), (sideX, sideY))
            batch = self.make_cutouts(TF.to_tensor(img).unsqueeze(0).to(device))
            embed = self.perceptor.encode_image(self.normalize(batch)).float()
            self.pMs.append(Prompt(embed, weight, stop).to(device))

        for seed, weight in zip(self.args.noise_prompt_seeds, self.args.noise_prompt_weights):
            gen = torch.Generator().manual_seed(seed)
            embed = torch.empty([1, self.perceptor.visual.output_dim]).normal_(generator=gen)
            self.pMs.append(Prompt(embed, weight).to(device))

        i = 0

        filename = filename.replace(" ", "_")
        if not path.exists(self.args.outdir):
          mkdir(self.args.outdir)
        # outname = self.set_valid_filename(self.args.outdir, filename, 0)
        outname = self.set_valid_dirname(self.args.outdir, filename, 0)

        saved_prompts_dir = path.join(self.args.outdir, "Saved_Prompts/")
        if not path.exists(saved_prompts_dir):
            mkdir(saved_prompts_dir)
        self.filelistpath = saved_prompts_dir + path.basename(outname) + ".txt"
        self.write_arg_list(prompts)
        try:
          with tqdm() as pbar:
            while True:
                self.train(i, outname)
                i += 1
                pbar.update()

        except KeyboardInterrupt:
            pass

    def set_sorted_folder(self, diroutname, filetype):
        diroutpath = path.join(self.content_output_path, diroutname)
        if not path.exists(diroutpath):
            # print("Creating sorted folder for " + str(diroutpath))
            mkdir(diroutpath)
        return diroutpath

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

    def synth(self):
        z_q = self.vector_quantize(self.z.movedim(1, 3), self.model.quantize.embedding.weight).movedim(3, 1)
        return ClampWithGrad.apply(self.model.decode(z_q).add(1).div(2), 0, 1)

    @torch.no_grad()
    def checkin(self, i, losses, name):
        losses_str = ', '.join(f'{loss.item():g}' for loss in losses)
        out = self.synth()
        sequence_number = i // self.args.display_freq

        outname = path.join(self.args.outdir, name)
        outname = self.image_output_path(outname, sequence_number=sequence_number)

        TF.to_pil_image(out[0].cpu()).save(outname)
        # stops the notebook file from getting too big by clearing the previous images from the output (they are still saved)
        if i > 0 and sequence_number % self.clear_interval == 0:
            clear_output()
        display.display(display.Image(str(outname)))
        tqdm.write(f'file: {path.basename(name)}, i: {i}, seq: {sequence_number}, loss: {sum(losses).item():g}, losses: {losses_str}')

    def ascend_txt(self):
        out = self.synth()
        iii = self.perceptor.encode_image(self.normalize(self.make_cutouts(out))).float()

        result = []

        if self.args.init_weight:
            result.append(F.mse_loss(self.z, self.z_orig) * self.args.init_weight / 2)

        for prompt in self.pMs:
            result.append(prompt(iii))

        return result

    def train(self, i, name):
        self.opt.zero_grad()
        lossAll = self.ascend_txt()
        if i % self.args.display_freq == 0:
            self.checkin(i, lossAll, name)

        loss = sum(lossAll)
        loss.backward()
        self.opt.step()
        with torch.no_grad():
            self.z.copy_(self.z.maximum(self.z_min).minimum(self.z_max))

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

    def set_valid_dirname(self, filename, basename, i):
        if i > 0:
            newname = "%s(%d)" % (basename, i)
        else:
            newname = basename

        unique_dir_name = True
        for root, dirs, files in walk(self.args.outdir):
            for dir in dirs:
                if path.basename(dir) == newname:
                    unique_dir_name = False
                    break

                dir_name = dir

            if not unique_dir_name:
                break

        if unique_dir_name:
            new_dir = path.join(path.dirname(dir_name), newname)
            mkdir(new_dir)
            return new_dir
          # return path.join(filename, newname)

        return self.set_valid_dirname(filename, basename, i + 1)

    def get_prompt_list(self, first, second, third, rest):
      param_list = [first, second, third]
      param_list = [p for p in param_list if p]
      prompt_list = param_list + rest
      return prompt_list

    def write_arg_list(self,args):
        start = """# Running this cell will generate images based on the form inputs ->
# It will also copy the contents of this cell and save it as a text file
# Copy the text from the file and paste it here to reuse the form inputs
from VQGAN_CLIP_Z_Quantize import VQGAN_CLIP_Z_Quantize
# If you want to add more text and image prompts,
# add them in a comma separated list in the brackets below
"""

        end = """VQGAN_CLIP_Z_Quantize(Other_txt_prompts,Other_img_prompts,Other_noise_seeds,Other_noise_weights,
Output_directory,Base_Image,Base_Image_Weight,Image_Prompt1,Image_Prompt2,Image_Prompt3,
Text_Prompt1,Text_Prompt2,Text_Prompt3,SizeX,SizeY,Noise_Seed_Number,Noise_Weight,Display_Frequency,Clear_Interval)"""

        comments = ["# (strings)",
          "# (strings of links or paths)",
          "# (longs)",
          "# (decimals)",
          "#@param {type:'string'}",
          "#@param {type:'string'}",
          "#@param {type:'slider', min:0, max:1, step:0.01}",
          "#@param {type:'string'}",
          "#@param {type:'string'}",
          "#@param {type:'string'}",
          "#@param {type:'string'}",
          "#@param {type:'string'}",
          "#@param {type:'string'}",
          "#@param {type:'number'}",
          "#@param {type:'number'}",
          "#@param {type:'string'}",
          "#@param {type:'slider', min:0, max:1, step:0.01}",
          "#@param {type:'integer'}",
          "#@param {type:'integer'}"]
        with open(self.filelistpath, "w", encoding="utf-8") as txtfile:
            i, txt = 0, ""
            for argname, argval in args.items():
                if comments[i] == "#@param {type:'string'}":
                    txt += f"{str(argname)}=\"{str(argval)}\" {comments[i]}"
                else:
                    txt += f"{str(argname)}={str(argval)} {comments[i]}"
                txt += "\n"
                i+=1
            print(f"writing settings to {self.filelistpath}")
            txtfile.write(start + txt + end)

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
