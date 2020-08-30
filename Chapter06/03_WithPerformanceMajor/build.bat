cls 

nvcc.exe -o BasicRayTracingCUDA.obj -c BasicRayTracing.cu

cl.exe /c /EHsc /I"C:\glew\include" BasicRayTracing.cpp

link.exe /OUT:"BasicRayTracing.exe" BasicRayTracing.obj BasicRayTracingCUDA.obj /LIBPATH:"C:\glew\lib\Release\x64" /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\lib\x64" cudart.lib user32.lib gdi32.lib kernel32.lib /SUBSYSTEM:WINDOWS

del BasicRayTracingCUDA.obj

del BasicRayTracing.obj

BasicRayTracing.exe
