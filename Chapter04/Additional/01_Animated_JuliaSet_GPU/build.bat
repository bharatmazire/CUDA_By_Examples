cls 

nvcc.exe -o JuliaSetCUDA.obj -c JuliaSet.cu

cl.exe /c /EHsc /I"C:\glew\include" JuliaSet.cpp

link.exe /OUT:"JuliaSet.exe" JuliaSet.obj JuliaSetCUDA.obj /LIBPATH:"C:\glew\lib\Release\x64" /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\lib\x64" cudart.lib user32.lib gdi32.lib kernel32.lib /SUBSYSTEM:WINDOWS

del JuliaSetCUDA.obj

del JuliaSet.obj

JuliaSet.exe
