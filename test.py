from os import path

Output_directory="/content/drive/MyDrive/VQGAN_Output" #@param {type:'string'}
Overwritten_Dir="/content/drive/MyDrive/VQGAN_Output/Psychedelic__Bright_Hair__Colorful_Glow(2)"
base_dir = path.join(Output_directory, path.basename(Overwritten_Dir))
base_dir_name = path.basename(base_dir)

# print(path.basename(path.join(Output_directory, path.basename(Overwritten_Dir))))
# print(path.basename(path.basename(Overwritten_Dir)))
print(base_dir_name)
