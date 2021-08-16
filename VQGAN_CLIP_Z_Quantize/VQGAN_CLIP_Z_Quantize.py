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
import json

class VQGAN_CLIP_Z_Quantize:
    def __init__(self, Other_txt_prompts,
                Other_img_prompts,
                Other_noise_seeds,
                Other_noise_weights,
                Output_directory,
                Base_Option, Base_Option_Weight,
                Image_Prompt1, Image_Prompt2, Image_Prompt3,
                Text_Prompt1,Text_Prompt2,Text_Prompt3,
                SizeX, SizeY,
                Noise_Seed_Number, Noise_Weight, Seed,
                Image_Model, CLIP_Model,
                Display_Frequency, Clear_Interval, Train_Iterations,
                Step_Size, Cut_N, Cut_Pow,
                Starting_Frame=None, Ending_Frame=None, Overwrite=False, Only_Save=False,
                Overwritten_Dir=None, Frame_Image=False):

        if not path.exists(Output_directory):
            mkdir(Output_directory)
        # prompts = OrderedDict()
        # prompts["Other_txt_prompts"] = Other_txt_prompts
        # prompts["Other_img_prompts"] = Other_img_prompts
        # prompts["Other_noise_seeds"] = Other_noise_seeds
        # prompts["Other_noise_weights"] = Other_noise_weights
        prompts = {"Other_txt_prompts": Other_txt_prompts,"Other_img_prompts": Other_img_prompts, "Other_noise_seeds": Other_noise_seeds, "Other_noise_weights": Other_noise_weights,
                    "Output_directory":Output_directory, "Base_Option":Base_Option, "Base_Option_Weight":Base_Option_Weight,
                    "Image_Prompt1":Image_Prompt1,"Image_Prompt2":Image_Prompt2,"Image_Prompt3":Image_Prompt3,
                    "Text_Prompt1":Text_Prompt1,"Text_Prompt2":Text_Prompt2,"Text_Prompt3":Text_Prompt3,
                    "SizeX":SizeX,"SizeY":SizeY,"Noise_Seed_Number":Noise_Seed_Number,
                    "Noise_Weight":Noise_Weight,"Seed":Seed, "Image_Model":Image_Model,"CLIP_Model":CLIP_Model,
                    "Display_Frequency":Display_Frequency,"Clear_Interval":Clear_Interval,
                    "Train_Iterations":Train_Iterations,"Step_Size":Step_Size,"Cut_N":Cut_N,"Cut_Pow":Cut_Pow,"Starting_Frame":Starting_Frame,
                    "Ending_Frame":Ending_Frame,"Only_Save":Only_Save,"Overwritten_Dir":Overwritten_Dir}
        # test_args = {"Starting_Frame":Starting_Frame,"Ending_Frame":Ending_Frame,"Overwrite":Overwrite,"Only_Save":Only_Save,"Overwritten_Dir":Overwritten_Dir}

        # prompts.update(arg_list)
        txt_prompts = self.get_prompt_list(Text_Prompt1, Text_Prompt2, Text_Prompt3, Other_txt_prompts)
        img_prompts = self.get_prompt_list(Image_Prompt1, Image_Prompt2, Image_Prompt3, Other_img_prompts)

        # Sets the filename based on the collection of Text_Prompts
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
        imgpath = None

        # if Base_Option exists, set the base directory to its final target or targets.
        if not Base_Option in (None, ""):

            sorted_imgs, txt_files = [], []

            is_frames  = False
            file_batch = False
            saved_prompts_dir = path.join(Output_directory, "Saved_Prompts/")

            # Setting the Base_Option to a directory will run each image and saved prompt text file in order.
            # Skips animated files but will run prompts that contain animated file parameters.
            if isdir(Base_Option):
                # base_dir = self.make_unique_dir(Output_directory, filename)
                base_dir = Output_directory

                base_dir_name = path.basename(base_dir)

                files = [f for f in listdir(saved_prompts_dir) if isfile(join(saved_prompts_dir, f))]
                args_basename = path.basename(Base_Option) + "_directory"

                file_batch = True
                files = [join(Base_Option, f) for f in listdir(Base_Option) if isfile(join(Base_Option, f))]

                # Separates images and text files to be run (will currently combine different image sets)
                # TODO: Separate sets of sorted images like in MLAnimator
                imgs = [f for f in files if path.splitext(f)[1] in ('.png', '.jpg')]
                txt_files = [f for f in files if path.splitext(f)[1] == '.txt']
                sorted_imgs = sorted(imgs, key=lambda f: self.get_file_num(f, len(imgs)))

            # Base_Options that are a path/URL to an animated file are separated into frames and ran individually.
            # Images are trained based on the amount of Train_Iterations.
            elif path.splitext(Base_Option)[1] in ('.mp4', '.gif'):
                base_dir = self.get_base_dir(Output_directory, filename, Overwritten_Dir=Overwritten_Dir)
                base_file_name = path.basename(path.splitext(Base_Option)[0])
                args_basename = path.basename(Base_Option) + "_directory"

                # args_file_name = base_dir_name + "_animation"

                is_frames = True
                file_batch = True

                split_frames_dirname = f"{base_file_name}_split_frames"

                frames_dir = join(base_dir, split_frames_dirname)
                if not exists(frames_dir):
                    mkdir(frames_dir)
                    imgname = f"{base_file_name}.%06d.png"
                    frames_dir_arg = path.join(frames_dir, imgname)
                    cmdargs = ['ffmpeg', '-i', Base_Option, frames_dir_arg]
                    subprocess.call(cmdargs)

                imgs = [join(frames_dir, f) for f in listdir(frames_dir) if isfile(join(frames_dir, f))]
                sorted_imgs = sorted(imgs, key=lambda f: self.get_file_num(f, len(imgs)))

            # Each run produces a text file of a JSON string. The file contains the settings for the run from which it was made.
            # Running the text file through the program will use the settings saved in it.
            elif path.splitext(Base_Option)[1] in ('.txt'):
                txt_files = [Base_Option]
                files = [f for f in listdir(saved_prompts_dir) if isfile(join(saved_prompts_dir, f))]
                args_basename = path.basename(Base_Option) + "_text"
            else:
                base_dir = self.get_base_dir(Output_directory, filename, Frame_Image=Frame_Image, Overwritten_Dir=Overwritten_Dir)
                print(f"Selecting Base_Option: {Base_Option}\nUsing filename: {filename}\nUsing base_dir: {base_dir}")
                base_dir_name = args_basename = path.basename(base_dir)

            ## TODO: make saved prompt text files save right
            args_file_name = self.set_valid_filename(files, saved_prompts_dir, args_basename, ".txt")
            self.write_args_file(Output_directory, args_file_name, prompts)
            if Only_Save:
                return
            imgLen = len(sorted_imgs)

            if file_batch:
                if imgLen > 0 and Train_Iterations > 0:

                    start, end = 1, imgLen
                    # If the option is an animated file, setting the Starting_Frame and Ending_Frame can limit from which frames to train.
                    # Be sure to use the Overwrite option to make frames if they are going in the same directory as other frame directories.
                    if is_frames:
                        try:
                            if Starting_Frame and Starting_Frame > 1 and Starting_Frame <= imgLen:
                                start = Starting_Frame
                            if Ending_Frame and Ending_Frame > 1 and Ending_Frame <= imgLen:
                                end = Ending_Frame

                            frameAmt = end - start
                            if frameAmt < 1:
                                start, end = 1, imgLen
                                print(f"Out of bounds frame selection, running through all {imgLen} frames.")

                        except:
                            start, end = 1, imgLen
                            print(f"Invalid frame selection, running through all {imgLen} frames.")



                    # j = start

                    print(f"start: {start}, end: {end}")
                    for img in sorted_imgs[start-1:end-1]:
                        imgname = path.basename(path.splitext(img)[0])

                        if is_frames:
                            img_base_dir = self.get_base_dir(Output_directory, imgname, Frame_Image=True)
                            target_dir = path.join(img_base_dir, f"{base_dir_name}_frame_{j}")
                        else:
                            target_dir = self.get_base_dir(Output_directory, imgname)
                        # j += 1
                        print(f"Going to target_dir: {target_dir}")
                        # if not exists(target_dir):
                        #     mkdir(target_dir)


                        vqgan = VQGAN_CLIP_Z_Quantize(Other_txt_prompts,Other_img_prompts,
                                    Other_noise_seeds,Other_noise_weights,target_dir,
                                    img, Base_Option_Weight,Image_Prompt1,Image_Prompt2,Image_Prompt3,
                                    Text_Prompt1,Text_Prompt2,Text_Prompt3,SizeX,SizeY,
                                    Noise_Seed_Number,Noise_Weight,Seed,Image_Model,CLIP_Model,
                                    Display_Frequency,Clear_Interval,Train_Iterations,Step_Size,Cut_N,Cut_Pow,
                                    Starting_Frame,Ending_Frame,Overwrite,Only_Save,Frame_Image=True)

                        if is_frames:
                            final_dir = path.join(base_dir, f"{base_dir_name}_final_frames")
                            print(f"Copying last frame to {final_dir}")
                            if not exists(final_dir):
                                mkdir(final_dir)

                            files = [f for f in listdir(final_dir) if isfile(join(final_dir, f))]
                            seq_num = int(len(files))+1
                            sequence_number_left_padded = str(seq_num).zfill(6)
                            newname = f"{base_dir_name}.{sequence_number_left_padded}.png"
                            final_out = path.join(final_dir, newname)
                            copyfile(vqgan.final_frame_path, final_out)

                # TODO: Change Saved_Prompts to JSON files rather than raw python code.
                # if len(txt_files) > 0:
                #     for f in txt_files:
                #         txt = open(f, "r")
                #         code= txt.read()
                #         txt.close()
                #         newfile = join(Output_directory, base_dir_name, path.basename(f) + ".py")
                #         py = open(newfile, "w")
                #         py.write(code)
                #         py.close()
                #         subprocess.call(["python", newfile])
                #         os.remove(newfile)
                if len(txt_files) > 0:
                    for f in txt_files:
                        txt = open(f, "r")
                        args = json.loads(txt.read())
                        VQGAN_CLIP_Z_Quantize(
                                    Other_txt_prompts=args.Other_txt_prompts,Other_img_prompts=args.Other_img_prompts, Other_noise_seeds=args.Other_noise_seeds, Other_noise_weights= args.Other_noise_weights,
                                    Output_directory=args.Output_directory, Base_Option=args.Base_Option, Base_Option_Weight=args.Base_Option_Weight,
                                    Image_Prompt1=args.Image_Prompt1,Image_Prompt2=args.Image_Prompt2,Image_Prompt3=args.Image_Prompt3,
                                    Text_Prompt1=args.Text_Prompt1,Text_Prompt2=args.Text_Prompt2,Text_Prompt3=args.Text_Prompt3,
                                    SizeX=args.SizeX,SizeY=args.SizeY, Noise_Seed_Number=args.Noise_Seed_Number,
                                    Noise_Weight=args.Noise_Weight,Seed=args.Seed,Image_Model=args.Image_Model,CLIP_Model=args.CLIP_Model,
                                    Display_Frequency=args.Display_Frequency,Clear_Interval=args.Clear_Interval,
                                    Train_Iterations=args.Train_Iterations,Step_Size=args.Step_Size,Cut_N=args.Cut_N,Cut_Pow=args.Cut_Pow,Starting_Frame=args.Starting_Frame,
                                    Ending_Frame=args.Ending_Frame,Only_Save=args.Only_Save,Overwritten_Dir=args.Overwritten_Dir)
                return
            else:
                if not Frame_Image:
                    self.write_args_file(Output_directory, filename, prompts)
                if Only_Save:
                    return

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
            outdir=Output_directory,
            init_image=Base_Option,
            init_weight=Base_Option_Weight,
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

        # This code belongs to the original VQGAN+CLIP (Z Quantize Method) notebook
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

        imgpath = self.get_pil_imagepath(Base_Option)
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

                # Main helper function for the training loop
                def train_and_update(i, last_image=False, retryTime=0):
                    try:
                        # new_filepath = self.train(i, Output_directory, last_image)
                        new_filepath = self.train(i, base_dir, last_image)

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
                if Train_Iterations > 0:
                    print(f"Begining training over {Train_Iterations} iterations.")
                    j = 0

                    while j < Train_Iterations - 1:
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

    def get_base_dir(self, Output_directory, filename, Frame_Image=False, Overwritten_Dir=None):
        make_unique_dir = True

        # If the rendered file is part of a batch of runs, place file in provided path
        if Frame_Image:
            base_dir, make_unique_dir = Output_directory, False

        # Overwritten_Dir is used to place files in a directory that already exists
        elif Overwritten_Dir:
            if not path.exists(Overwritten_Dir):
                print("Directory to overwrite doesn't exist, creating new directory to avoid overwriting unintended directory.")

            else:
                base_dir = join(Output_directory, path.basename(Overwritten_Dir))
                make_unique_dir = False

        # Not overwriting will make the filename unique and make a new directory for its files.
        if make_unique_dir: base_dir = self.make_unique_dir(Output_directory, filename)

        return base_dir

    def make_unique_dir(self, Output_directory, filename):
        dirs = [x[0] for x in walk(Output_directory)]
        return self.set_valid_dirname(dirs, Output_directory, filename)

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
        return 0

    @torch.no_grad()
    def checkin(self, i, losses, outpath):
        losses_str = ', '.join(f'{loss.item():g}' for loss in losses)
        out = self.synth()
        sequence_number = i // self.args.display_freq

        # TODO: change to display same name as directory
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
            print(f"checkin on: {outpath}")
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
        # split = path.splitext(output_path)[0]
        # parent = path.dirname(output_path)
        base = path.basename(output_path)
        print(f"""Main image_output_path: {output_path}\n
                base of output_path: {base}""")
        if sequence_number:
            sequence_number_left_padded = str(sequence_number).zfill(6)
            newname = f"{base}.{sequence_number_left_padded}"
        else:
            newname = base
        output_path = path.join(output_path, newname)
        return Path(f"{output_path}.png")

    def write_args_file(self, out, base, prompts):
        saved_prompts_dir = path.join(out, "Saved_Prompts/")
        if not path.exists(saved_prompts_dir):
            mkdir(saved_prompts_dir)

        # TODO: change this to CSV, JSON, or XML
        self.filelistpath = saved_prompts_dir + base + ".txt"
        self.write_arg_list(prompts)


    def set_valid_dirname(self, dirs, out, basename, i=0):
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

    def set_valid_filename(self, files, out, basename, ext, i=0):
        if i > 0:
            newname = "%s(%d).%s" % (basename, i, ext)
        else:
            newname = basename

        unique_file_name = True

        if len(files) < 1:
            # new_path = path.join(out, newname)
            # mkdir(new_path)
            return newname

        for file in files:
            # if path.basename(file) == newname:
            # bname = path.basename(file)
            print(f"checking: {path.basename(file)} against: {newname}")
            if path.basename(file) == newname:

                unique_file_name = False
                break

        if unique_file_name:
            # new_path = path.join(out, newname)
            #
            # mkdir(new_path)
            return newname

        return self.set_valid_filename(files, out, basename, ext, i + 1)

    def get_prompt_list(self, first, second, third, rest):
      param_list = [first, second, third]
      param_list = [p for p in param_list if p]
      prompt_list = param_list + rest
      return prompt_list

    def write_arg_list(self,args):
        with open(self.filelistpath, "w", encoding="utf-8") as txtfile:

            json_args = json.dumps(args)
            print(f"writing settings to {self.filelistpath}")
            txtfile.write(json_args)

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
