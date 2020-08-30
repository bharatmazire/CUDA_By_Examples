cls 

nvcc.exe -o SharedMemoryBitmapCUDA.obj -c SharedMemoryBitmap.cu

cl.exe /c /EHsc /I"C:\glew\include" SharedMemoryBitmap.cpp

link.exe /OUT:"SharedMemoryBitmap.exe" SharedMemoryBitmap.obj SharedMemoryBitmapCUDA.obj /LIBPATH:"C:\glew\lib\Release\x64" /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\lib\x64" cudart.lib user32.lib gdi32.lib kernel32.lib /SUBSYSTEM:WINDOWS

del SharedMemoryBitmapCUDA.obj

del SharedMemoryBitmap.obj

SharedMemoryBitmap.exe
