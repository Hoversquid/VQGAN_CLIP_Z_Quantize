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
from os import chdir, mkdir, path, getcwd, walk, listdir
from os.path import isfile, isdir, exists, join
from CLIP import clip
from IPython.display import clear_output
from collections import OrderedDict
from shutil import copyfile
import subprocess

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
                Image_Model, CLIP_Model,
                Display_Frequency, Clear_Interval, Max_Iterations,
                Step_Size, Cut_N, Cut_Pow,
                Starting_Frame=None, Ending_Frame=None, Overwrite=False, Only_Save=False, Is_Frame=False,
                Overwritten_Dir=None):

        if not path.exists(Output_directory):
            mkdir(Output_directory)
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
                    "Noise_Weight":Noise_Weight,"Seed":Seed,
                    "Image_Model":Image_Model,"CLIP_Model":CLIP_Model,
                    "Display_Frequency":Display_Frequency,"Clear_Interval":Clear_Interval,"Max_Iterations":Max_Iterations,"Step_Size":Step_Size,"Cut_N":Cut_N,"Cut_Pow":Cut_Pow}
        test_args = {"Starting_Frame":Starting_Frame,"Ending_Frame":Ending_Frame,"Overwrite":Overwrite,"Only_Save":Only_Save}
        prompts.update(arg_list)
        txt_prompts = self.get_prompt_list(Text_Prompt1, Text_Prompt2, Text_Prompt3, Other_txt_prompts)
        img_prompts = self.get_prompt_list(Image_Prompt1, Image_Prompt2, Image_Prompt3, Other_img_prompts)

        filename = ""
        name_limit = 42
        for i, prompt in enumerate(txt_prompts):
            name_length = name_limit - len(filename)
            if name_length > 0:
              filename += prompt[:name_length]
              if len(filename) + 2 < name_limit and i + 1 < len(txt_prompts):
                filename += "__"

        if filename == "":
          filename = "No_Prompts"

        filename = filename.replace(" ", "_")
        dirs = [x[0] for x in walk(Output_directory)]

        if Is_Frame:
            outpath = Output_directory
        elif Overwrite:
            if Overwritten_Dir:
                outpath = Overwritten_Dir
                base_dir = path.basename(Overwritten_Dir)
            else:
                outpath = Output_directory
                base_dir = path.basename(Overwritten_Dir)
        else:
            outpath = self.set_valid_dirname(dirs, Output_directory, filename, 0)
        imgpath = None

        base_out = path.basename(outpath)

        if not Base_Image in (None, ""):
            sorted_imgs = []

            if isdir(Base_Image):
                files = [join(Base_Image, f) for f in listdir(Base_Image) if isfile(join(Base_Image, f))]
                imgs = [f for f in files if path.splitext(f)[1] in ('.png', '.jpg')]
                txt_files = [f for f in files if path.splitext(f)[1] == '.txt']
                sorted_imgs = sorted(imgs, key=lambda f: self.get_file_num(f, len(imgs)))
                base_name = path.basename(Base_Image)

            elif path.splitext(Base_Image)[1] in ('.mp4', '.gif'):
                base_name = path.basename(path.splitext(Base_Image)[0])
                split_frames_dirname = f"{base_name}_split_frames"
                frames_dir = join(base_dir, split_frames_dirname)
                if not exists(frames_dir):
                    mkdir(frames_dir)
                    imgname = f"{base_name}.%06d.png"
                    frames_dir_arg = path.join(frames_dir, imgname)
                    cmdargs = ['ffmpeg', '-i', Base_Image, frames_dir_arg]
                    subprocess.call(cmdargs)

                imgs = [join(frames_dir, f) for f in listdir(frames_dir) if isfile(join(frames_dir, f))]
                sorted_imgs = sorted(imgs, key=lambda f: self.get_file_num(f, len(imgs)))

            imgLen = len(sorted_imgs)
            start, end = 1, imgLen

            try:
                if Starting_Frame > 1 and Starting_Frame <= imgLen:
                    start = Starting_Frame
                if Ending_Frame > 1 and Ending_Frame <= imgLen:
                    end = Ending_Frame
                if end - start < 1:
                    start, end = 1, imgLen

            except:
                print("Invalid frame selection")
                start, end = 1, imgLen


            if len(sorted_imgs) > 0 and Max_Iterations > 0:
                self.write_args_file(Output_directory, base_out, prompts, test_args)
                if Only_Save:
                    return
                j = start

                for img in sorted_imgs[start-1:end-1]:
                    dir_name = f"{base_out}_frame_{j}"
                    j += 1
                    new_frame_dir = path.join(base_dir, dir_name)
                    if not exists(new_frame_dir):
                        mkdir(new_frame_dir)
                    vqgan = VQGAN_CLIP_Z_Quantize(Other_txt_prompts,Other_img_prompts,
                                Other_noise_seeds,Other_noise_weights,new_frame_dir,
                                img, Base_Image_Weight,Image_Prompt1,Image_Prompt2,Image_Prompt3,
                                Text_Prompt1,Text_Prompt2,Text_Prompt3,SizeX,SizeY,
                                Noise_Seed_Number,Noise_Weight,Seed,Image_Model,CLIP_Model,
                                Display_Frequency,Clear_Interval,Max_Iterations,Step_Size,Cut_N,Cut_Pow,
                                Starting_Frame,Ending_Frame,Overwrite,Only_Save,Is_Frame=True)

                    final_frame_dir_name = f"{base_out}_final_frames"
                    final_dir = path.join(base_dir, final_frame_dir_name)
                    print(f"Copying last frame to {final_dir}")
                    if not exists(final_dir):
                        mkdir(final_dir)

                    files = [f for f in listdir(final_dir) if isfile(join(final_dir, f))]
                    seq_num = int(len(files))+1
                    sequence_number_left_padded = str(seq_num).zfill(6)
                    newname = f"{base_out}.{sequence_number_left_padded}.png"
                    final_out = path.join(final_dir, newname)
                    copyfile(vqgan.final_frame_path, final_out)

            if len(txt_files) > 0:
                for f in txt_files:
                    txt = open(f, "r")
                    code= txt.read()
                    txt.close()
                    newfile = join(Output_directory, base_out, path.basename(f) + ".py")
                    py = open(newfile, "w")
                    py.write(code)
                    py.close()
                    subprocess.call(["python", newfile])
                    os.remove(newfile)

            return
        else:
            if not Is_Frame:
                self.write_args_file(Output_directory, base_out, prompts)
            if Only_Save:
                return

            imgpath = self.get_pil_imagepath(Base_Image)

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

        self.args = argparse.Namespace(
            outdir=Output_directory, # this is the name of where your output will go
            init_image=Base_Image,
            init_weight=Base_Image_Weight,
            prompts=txt_prompts,
            image_prompts=img_prompts,
            noise_prompt_seeds=noise_prompt_seeds,
            noise_prompt_weights=noise_prompt_weights,
            size=[SizeX, SizeY],
            clip_model=CLIP_Model,
            vqgan_config=f'{Image_Model}.yaml',
            vqgan_checkpoint=f'{Image_Model}.ckpt',
            step_size=Step_Size,
            cutn=Cut_N,
            cut_pow=Cut_Pow,
            display_freq=Display_Frequency,
            seed=Seed)

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

        for i, prompt in enumerate(self.args.prompts):
            txt, weight, stop = self.parse_prompt(prompt)
            embed = self.perceptor.encode_text(clip.tokenize(txt).to(device)).float()
            self.pMs.append(Prompt(embed, weight, stop).to(device))

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

        try:
            with tqdm() as pbar:

                def train_and_update(i, outpath=outpath, last_image=False, retryTime=0):
                    try:
                        new_filepath = self.train(i, outpath, last_image)
                        pbar.update()
                        return new_filepath

                    except RuntimeError:
                        print("RuntimeError: " + sys.exc_info()[0])
                        if retryTime>0:
                            print(f"Retrtying in {retryTime}.\nYou may need to lower your size settings or change models.")
                        torch.cuda.empty_cache()
                        time.sleep(retryTime)
                        train_and_update(i, output_path=output_path, last_image=last_image, retryTime=retryTime+3)


                # Set to -1 to run forever
                if Max_Iterations > 0:
                    j = 0

                    while j < Max_Iterations - 1:
                        last_frame_path = train_and_update(i)
                        i += 1
                        j += 1

                    self.final_frame_path = train_and_update(i, last_image=True)
                    torch.cuda.empty_cache()

                else:
                    while True:
                        train_and_update(i)
                        i += 1

        except KeyboardInterrupt:
            # torch.cuda.empty_cache()
            pass

    def set_sorted_folder(self, diroutname, filetype):
        diroutpath = path.join(self.content_output_path, diroutname)
        if not path.exists(diroutpath):
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

    def get_file_num(self, f, lastnum):
        namestr = f.split(".")
        if namestr[-2].isnumeric():
            return int(namestr[-2])

    @torch.no_grad()
    def checkin(self, i, losses, outpath):
        losses_str = ', '.join(f'{loss.item():g}' for loss in losses)
        out = self.synth()
        sequence_number = i // self.args.display_freq

        outname = self.image_output_path(outpath, sequence_number=sequence_number)
        TF.to_pil_image(out[0].cpu()).save(outname)
        # stops the notebook file from getting too big by clearing the previous images from the output
        # (they are still saved)
        if i > 0 and sequence_number % self.clear_interval == 0:
            clear_output()
        display.display(display.Image(str(outname)))
        tqdm.write(f'file: {path.basename(outpath)}, i: {i}, seq: {sequence_number}, loss: {sum(losses).item():g}, losses: {losses_str}')
        return outname

    def ascend_txt(self):
        out = self.synth()
        iii = self.perceptor.encode_image(self.normalize(self.make_cutouts(out))).float()

        result = []

        if self.args.init_weight:
            result.append(F.mse_loss(self.z, self.z_orig) * self.args.init_weight / 2)

        for prompt in self.pMs:
            result.append(prompt(iii))

        return result

    def train(self, i, outpath, override=False):
        self.opt.zero_grad()
        lossAll = self.ascend_txt()
        if i % self.args.display_freq == 0 or override:
            new_filepath = self.checkin(i, lossAll, outpath)
        else: new_filepath = None

        loss = sum(lossAll)
        loss.backward()
        self.opt.step()
        with torch.no_grad():
            self.z.copy_(self.z.maximum(self.z_min).minimum(self.z_max))

        return new_filepath

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
        split = path.splitext(output_path)[0]
        base = path.basename(output_path)
        if sequence_number:
            sequence_number_left_padded = str(sequence_number).zfill(6)
            newname = f"{base}.{sequence_number_left_padded}"
        else:
            newname = base
        output_path = path.join(split, newname)
        return Path(f"{output_path}.png")

    def write_args_file(self, out, base, prompts, test_args):
        saved_prompts_dir = path.join(out, "Saved_Prompts/")
        if not path.exists(saved_prompts_dir):
            mkdir(saved_prompts_dir)
        self.filelistpath = saved_prompts_dir + base + ".txt"
        self.write_arg_list(prompts, test_args)


    def set_valid_dirname(self, dirs, out, basename, i):
        if i > 0:
            newname = "%s(%d)" % (basename, i)
        else:
            newname = basename

        unique_dir_name = True

        if len(dirs) < 1:
            new_path = path.join(out, newname)
            mkdir(new_path)
            return new_path
        for dir in dirs:
            if path.basename(dir) == newname:
                unique_dir_name = False
                break

        if unique_dir_name:
            new_path = path.join(out, newname)

            mkdir(new_path)
            return new_path

        return self.set_valid_dirname(dirs, out, basename, i + 1)

    def get_prompt_list(self, first, second, third, rest):
      param_list = [first, second, third]
      param_list = [p for p in param_list if p]
      prompt_list = param_list + rest
      return prompt_list

    def write_arg_list(self,args,test_args):
        start = """# Running this cell will generate images based on the form inputs ->
# It will also copy the contents of this cell and save it as a text file
# Copy the text from the file and paste it here to reuse the form inputs
from VQGAN_CLIP_Z_Quantize import VQGAN_CLIP_Z_Quantize
# If you want to add more text and image prompts,
# add them in a comma separated list in the brackets below
"""

        end = """VQGAN_CLIP_Z_Quantize(Other_txt_prompts,Other_img_prompts,Other_noise_seeds,Other_noise_weights,
Output_directory,Base_Image,Base_Image_Weight,Image_Prompt1,Image_Prompt2,Image_Prompt3,
Text_Prompt1,Text_Prompt2,Text_Prompt3,SizeX,SizeY,Noise_Seed_Number,Noise_Weight,Seed,Image_Model,CLIP_Model,Display_Frequency,Clear_Interval,Max_Iterations,Step_Size,Cut_N,Cut_Pow,Starting_Frame,Ending_Frame)"""

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
          "#@param ['drive/MyDrive/colab/coco', 'vqgan_imagenet_f16_1024', 'vqgan_imagenet_f16_16384', 'coco', 'wikiart_16384', 'wikiart_1024', 'sflickr', 'faceshq']",
          "#@param ['RN50', 'RN101', 'RN50x4', 'ViT-B/32']",
          "#@param {type:'integer'}",
          "#@param {type:'string'}",
          "#@param {type:'integer'}",
          "#@param {type:'number'}",
          "#@param {type:'number'}",
          "#@param {type:'number'}",
          ]
        with open(self.filelistpath, "w", encoding="utf-8") as txtfile:
            i, txt = 0, ""
            for argname, argval in args.items():
                if comments[i].startswith("#@param {type:'string'}", "#@param ["):
                    txt += f"{str(argname)}=\"{str(argval)}\" {comments[i]}"
                else:
                    txt += f"{str(argname)}={str(argval)} {comments[i]}"
                txt += "\n"
                i+=1

            for argname, argval in test_args.items():
                txt += f"{str(argname)}={str(argval)}\n"
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
