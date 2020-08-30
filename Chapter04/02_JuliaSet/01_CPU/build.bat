cls

cl.exe /c /EHsc /I"C:\glew\include" JuliaSet.cpp


link.exe /LIBPATH:"C:\glew\lib\Release\x64" JuliaSet.obj user32.lib gdi32.lib kernel32.lib opengl32.lib /MACHINE:X64

del JuliaSet.obj

JuliaSet.exe
