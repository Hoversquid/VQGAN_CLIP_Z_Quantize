# Running this cell will generate images based on the form inputs ->
# It will also copy the contents of this cell and save it as a text file
# Copy the text from the file and paste it here to reuse the form inputs
from VQGAN_CLIP_Z_Quantize import VQGAN_CLIP_Z_Quantize
# If you want to add more text and image prompts,
# add them in a comma separated list in the brackets below
Other_txt_prompts=[] # (strings)
Other_img_prompts=[] # (strings of links or paths)
Other_noise_seeds=[] # (longs)
Other_noise_weights=[] # (decimals)
Output_directory="/content/drive/MyDrive/VQGAN_Output" #@param {type:'string'}
Base_Image="/content/drive/MyDrive/jchan_smile_swing.mp4" #@param {type:'string'}
Base_Image_Weight=0.68 #@param {type:'slider', min:0, max:1, step:0.01}
Image_Prompt1="" #@param {type:'string'}
Image_Prompt2="" #@param {type:'string'}
Image_Prompt3="" #@param {type:'string'}
Text_Prompt1="Psychedelic" #@param {type:'string'}
Text_Prompt2="Bright Hair" #@param {type:'string'}
Text_Prompt3="Colorful Glow" #@param {type:'string'}
SizeX=600 #@param {type:'number'}
SizeY=713 #@param {type:'number'}
Noise_Seed_Number="" #@param {type:'string'}
Noise_Weight=0.41 #@param {type:'slider', min:0, max:1, step:0.01}
Seed=0 #@param {type:'integer'}
Image_Model="vqgan_imagenet_f16_16384" #@param ['drive/MyDrive/colab/coco', 'vqgan_imagenet_f16_1024', 'vqgan_imagenet_f16_16384', 'coco', 'wikiart_16384', 'wikiart_1024', 'sflickr', 'faceshq']
CLIP_Model="ViT-B/32" #@param ['RN50', 'RN101', 'RN50x4', 'ViT-B/32']
Display_Frequency=10 #@param {type:'integer'}
Clear_Interval="50" #@param {type:'string'}
Max_Iterations=600 #@param {type:'integer'}
Step_Size=0.04 #@param {type:'number'}
Cut_N=80 #@param {type:'number'}
Cut_Pow=1.0 #@param {type:'number'}
Starting_Frame=6
Overwrite=True
VQGAN_CLIP_Z_Quantize(Other_txt_prompts,Other_img_prompts,Other_noise_seeds,Other_noise_weights,
Output_directory,Base_Image,Base_Image_Weight,Image_Prompt1,Image_Prompt2,Image_Prompt3,
Text_Prompt1,Text_Prompt2,Text_Prompt3,SizeX,SizeY,Noise_Seed_Number,Noise_Weight,Seed,Image_Model,CLIP_Model,Display_Frequency,Clear_Interval,Max_Iterations,Step_Size,Cut_N,Cut_Pow,Starting_Frame,Overwrite=Overwrite)
