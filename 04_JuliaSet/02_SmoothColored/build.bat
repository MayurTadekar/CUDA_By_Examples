cls

cl /EHsc /c OGL.cpp /I "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\include"

nvcc -c julia.cu

rc OGL.rc

link OGL.obj julia.obj OGL.res opengl32.lib user32.lib gdi32.lib glew32.lib /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\lib\x64" cudart.lib /SUBSYSTEM:WINDOWS 

@REM  OGL.exe

