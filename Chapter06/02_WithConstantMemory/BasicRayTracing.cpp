#include<windows.h>
#include<stdio.h>
#include<GL/glew.h>
#include<gl/GL.h>
#define _USE_MATH_DEFINES 1
#include<math.h>
#include"vmath.h"

// for libraries
#pragma comment(lib,"glew32.lib")
#pragma comment(lib,"opengl32.lib")
#pragma comment(lib,"cudart.lib")

#define WIN_WIDTH 800
#define WIN_HEIGHT 600

#define CHECK_IMAGE_WIDTH 2048
#define CHECK_IMAGE_HEIGHT 2048

using namespace vmath;

// declaration of global function
LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

// declaration of global variables
HWND			ghwnd;
bool			bIsFullScreen = false;
DWORD			dwStyle;
WINDOWPLACEMENT wpPrev = { sizeof(WINDOWPLACEMENT) };

// for openGL
HDC				ghdc = NULL;
HGLRC			ghrc = NULL;
bool			gbActiveWindow = false;

// for file io
FILE* gpFile = NULL;


// enum
enum
{
	AMC_ATTRIBUTE_POSITION = 0,
	AMC_ATTRIBUTE_COLOR,
	AMC_ATTRIBUTE_NORMAL,
	AMC_ATTRIBUTE_TEXCOORD0
};

// global variables
GLuint gShaderProgramObject;

GLuint vao_rectangle;

GLuint vbo_position_rectangle;
GLuint vbo_texture_rectangle;

GLuint mvpUniform;
GLuint samplerUniform;

mat4 PerspectiveProjectionMatrix;


//GLubyte CheckImage[CHECK_IMAGE_WIDTH][CHECK_IMAGE_HEIGHT][4];
GLubyte CheckImage[CHECK_IMAGE_WIDTH * CHECK_IMAGE_HEIGHT * 4];			// GLubyte = unsigned char
GLuint texImage;

// main
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow)
{
	// declaration of functions
	int initialize(void);
	void display(void);
	void update(void);
	void ToggleFullScreen(void);

	// declaration of variables
	MSG			msg;
	WNDCLASSEX	wndclass;
	TCHAR		szAppName[] = TEXT("CUDA Basic RayTracer - with constant memory");
	HWND		hwnd;

	bool		bDone = false;
	int			iRet = 0;

	// code
	if (fopen_s(&gpFile, "Log.txt", "w") != 0)
	{
		MessageBox(NULL, TEXT("Unable to create Log File !!"), TEXT("ERROR "), MB_OK | MB_ICONERROR);
		exit(0);
	}
	else
	{
		//MessageBox(NULL, TEXT("Successfully created Log File !!"), TEXT("SUCCESS "), MB_OK);
	}

	// code
	wndclass.cbClsExtra = 0;
	wndclass.cbSize = sizeof(WNDCLASSEX);
	wndclass.cbWndExtra = 0;
	wndclass.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
	wndclass.hCursor = LoadCursor(NULL, IDC_ARROW);
	wndclass.hIcon = LoadIcon(NULL, IDI_APPLICATION);
	wndclass.hIconSm = LoadIcon(NULL, IDI_APPLICATION);
	wndclass.hInstance = hInstance;
	wndclass.lpfnWndProc = WndProc;
	wndclass.lpszClassName = szAppName;
	wndclass.lpszMenuName = NULL;
	wndclass.style = CS_VREDRAW | CS_HREDRAW | CS_OWNDC;

	if (!RegisterClassEx(&wndclass))
	{
		MessageBox(NULL, TEXT("ERROR : Unable to Register Class !!"), TEXT("ERROR"), MB_OK | MB_ICONERROR);
		exit(0);
	}

	hwnd = CreateWindowEx(WS_EX_APPWINDOW,
		szAppName,
		TEXT("Basic RayTracer - with constant memory"),
		WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN | WS_CLIPSIBLINGS | WS_VISIBLE,
		100,
		100,
		WIN_WIDTH,
		WIN_HEIGHT,
		NULL,
		NULL,
		hInstance,
		NULL);

	ghwnd = hwnd;

	iRet = initialize();
	if (iRet == -1)
	{
		fprintf(gpFile, "ChoosePixelFormat() Fail \n");
		fflush(gpFile);
		DestroyWindow(hwnd);
	}
	else if (iRet == -2)
	{
		fprintf(gpFile, "SetPixelFormat() Fail \n");
		fflush(gpFile);
		DestroyWindow(hwnd);
	}
	else if (iRet == -3)
	{
		fprintf(gpFile, "wglCreateContext() Fail \n");
		fflush(gpFile);
		DestroyWindow(hwnd);
	}
	else if (iRet == -4)
	{
		fprintf(gpFile, "wglMakeCurrent() Fail \n");
		fflush(gpFile);
		DestroyWindow(hwnd);
	}
	else if (iRet == -5)
	{
		fprintf(gpFile, "glewInit() Fail \n");
		fflush(gpFile);
		DestroyWindow(hwnd);
	}
	else if (iRet == -6)
	{
		fprintf(gpFile, "Some Compile time error occurs at Vertex Shader \n");
		fflush(gpFile);
		DestroyWindow(hwnd);
	}
	else if (iRet == -7)
	{
		fprintf(gpFile, "Some Compile time error occurs at Fragment Shader \n");
		fflush(gpFile);
		DestroyWindow(hwnd);
	}
	else if (iRet == -8)
	{
		fprintf(gpFile, "Some Link time error occurs at Shader Program linking time \n");
		fflush(gpFile);
		DestroyWindow(hwnd);
	}
	else if (iRet == -9)
	{
		fprintf(gpFile, "Load Texture Fails \n");
		fflush(gpFile);
		DestroyWindow(hwnd);
	}
	else
	{
		fprintf(gpFile, "Initialization Successed \n");
		fflush(gpFile);
	}


	ShowWindow(hwnd, iCmdShow);
	SetForegroundWindow(hwnd);
	SetFocus(hwnd);
	ToggleFullScreen();

	// game loop
	while (bDone == false)
	{
		if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
		{
			if (msg.message == WM_QUIT)
			{
				bDone = true;
			}
			else
			{
				TranslateMessage(&msg);
				DispatchMessage(&msg);
			}
		}
		else
		{
			if (gbActiveWindow == true)
			{
				update();
			}
			display();
		}
	}

	return((int)msg.wParam);
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam)
{
	// declaration of function
	void ToggleFullScreen(void);
	void resize(int, int);
	//void display(void);
	void uninitialize(void);

	// code
	switch (iMsg)
	{
	case WM_CREATE:
		fprintf(gpFile, "In WM_CREATE \n");
		fflush(gpFile);
		break;

	case WM_SETFOCUS:
		fprintf(gpFile, "In WM_SETFOCUS \n");
		fflush(gpFile);
		gbActiveWindow = true;
		break;

	case WM_KILLFOCUS:
		fprintf(gpFile, "In WM_KILLFOCUS \n");
		fflush(gpFile);
		gbActiveWindow = false;
		break;

	case WM_SIZE:
		fprintf(gpFile, "In WM_SIZE \n");
		fflush(gpFile);
		resize(LOWORD(lParam), HIWORD(lParam));
		break;

	case WM_ERASEBKGND:
		return(0);				// return(0) : so that it will go to else part in game loop( to call display() ), instead of DefWindowProc()

	case WM_CLOSE:
		fprintf(gpFile, "IN WM_CLOSE \n");
		fflush(gpFile);
		DestroyWindow(hwnd);
		break;

	case WM_KEYDOWN:
		fprintf(gpFile, "In WM_KEYDOWN \n");
		fflush(gpFile);
		switch (wParam)
		{
		case VK_ESCAPE:
			DestroyWindow(hwnd);
			break;

		case 0x46:
		case 'f':
			ToggleFullScreen();
			//UpdateWindow(hwnd);
			break;
		}
		break;

	case WM_DESTROY:
		fprintf(gpFile, "In WM_DESTROY \n");
		fflush(gpFile);
		uninitialize();
		PostQuitMessage(0);
		break;
	}

	return(DefWindowProc(hwnd, iMsg, wParam, lParam));
}

void ToggleFullScreen(void)
{

	fprintf(gpFile, "In : ToggleFullScreen() \n");
	fflush(gpFile);

	// declataion of variable
	MONITORINFO mi;

	// code
	if (bIsFullScreen == false)
	{
		dwStyle = GetWindowLong(ghwnd, GWL_STYLE);
		if (dwStyle & WS_OVERLAPPEDWINDOW)
		{
			mi = { sizeof(MONITORINFO) };
			if (GetWindowPlacement(ghwnd, &wpPrev) && GetMonitorInfo(MonitorFromWindow(ghwnd, MONITORINFOF_PRIMARY), &mi))
			{
				SetWindowLong(ghwnd, GWL_STYLE, dwStyle & ~WS_OVERLAPPEDWINDOW);
				SetWindowPos(ghwnd,
					HWND_TOP,
					mi.rcMonitor.left,
					mi.rcMonitor.top,
					(mi.rcMonitor.right - mi.rcMonitor.left),
					(mi.rcMonitor.bottom - mi.rcMonitor.top),
					SWP_NOZORDER | SWP_FRAMECHANGED);
			}
		}
		ShowCursor(FALSE);
		bIsFullScreen = true;

	}
	else
	{
		SetWindowLong(ghwnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(ghwnd, &wpPrev);
		SetWindowPos(ghwnd,
			HWND_TOP,
			0,
			0,
			0,
			0,
			SWP_NOZORDER | SWP_FRAMECHANGED | SWP_NOMOVE | SWP_NOSIZE | SWP_NOOWNERZORDER);
		ShowCursor(TRUE);
		bIsFullScreen = false;

	}
	fprintf(gpFile, "OUT : ToggleFullScreen() \n");
	fflush(gpFile);
}

int initialize()
{
	fprintf(gpFile, "IN : initialize() \n");
	fflush(gpFile);

	// function declaration
	void resize(int, int);
	void loadTexture(void);

	// variables declarations
	GLuint gVertexShaderObject;
	GLuint gFragmentShaderObject;


	PIXELFORMATDESCRIPTOR pfd;
	int iPixelFormatIndex;

	GLenum result;

	// variables for error checking 
	GLint iShaderCompileStatus = 0;
	GLint iInfoLogLength = 0;
	GLchar* szInfoLog = NULL;
	GLint iProgramLinkStatus = 0;

	// code
#pragma region BASIC_OGL_PART

	// initialize pfd structure
	ZeroMemory(&pfd, sizeof(PIXELFORMATDESCRIPTOR));
	//else use
	// memset((void *)&pfd,NULL,sizeof(PIXELFORMATDESCRIPTOR));

	pfd.nSize = sizeof(PIXELFORMATDESCRIPTOR);
	pfd.nVersion = 1;
	pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
	pfd.iPixelType = PFD_TYPE_RGBA;								// PFG_RGBA ;
	pfd.cColorBits = 32;
	pfd.cRedBits = 8;
	pfd.cGreenBits = 8;
	pfd.cBlueBits = 8;
	pfd.cAlphaBits = 8;
	pfd.cDepthBits = 32;			// for depth (3D)

	ghdc = GetDC(ghwnd);

	iPixelFormatIndex = ChoosePixelFormat(ghdc, &pfd);
	if (iPixelFormatIndex == 0)
	{
		return(-1);
	}

	if (SetPixelFormat(ghdc, iPixelFormatIndex, &pfd) == FALSE)
	{
		return(-2);
	}

	ghrc = wglCreateContext(ghdc);
	if (ghrc == NULL)
	{
		return(-3);
	}

	if (wglMakeCurrent(ghdc, ghrc) == FALSE)
	{
		return(-4);
	}

	result = glewInit();
	if (result != GLEW_OK)
	{
		return(-5);
	}

#pragma endregion


#pragma region VERETX_SHADER
	fprintf(gpFile, "IN : initialize() FOR VERTEX SHADER \n");
	fflush(gpFile);
	// vertex shader part start

	// define vertex shader object
	gVertexShaderObject = glCreateShader(GL_VERTEX_SHADER);

	fprintf(gpFile, "IN : initialize() done upto glCreateShader() \n");
	fflush(gpFile);

	// write vertex shader code
	const GLchar* vertexShaderSourceCode =
		"#version 450 core" \
		"\n" \
		"in vec4 vPosition;" \
		"in vec2 vTexcoord;" \
		"out vec2 out_texcoord;" \
		"uniform mat4 u_mvp_matrix;" \
		"void main(void)" \
		"{" \
		"gl_Position = u_mvp_matrix * vPosition;" \
		"out_texcoord = vTexcoord;" \
		"}";

	// specify above source, to vertex shader object
	glShaderSource(gVertexShaderObject, 1, (GLchar**)&vertexShaderSourceCode, NULL);

	fprintf(gpFile, "IN : initialize() done upto glShaderSource() \n");
	fflush(gpFile);

	// compile the vertex shader
	glCompileShader(gVertexShaderObject);

	fprintf(gpFile, "IN : initialize() done upto glCompileShader() \n");
	fflush(gpFile);

	// error checking for vertex shader
	iShaderCompileStatus = 0;
	iInfoLogLength = 0;
	szInfoLog = NULL;

	glGetShaderiv(gVertexShaderObject, GL_COMPILE_STATUS, &iShaderCompileStatus);

	fprintf(gpFile, "IN : initialize() done upto 1st glGetShaderiv() \n");
	fflush(gpFile);

	if (iShaderCompileStatus == GL_FALSE)
	{
		glGetShaderiv(gVertexShaderObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0)
		{
			szInfoLog = (GLchar*)malloc(iInfoLogLength);
			if (szInfoLog != NULL)
			{
				GLsizei written;
				glGetShaderInfoLog(gVertexShaderObject, iInfoLogLength, &written, szInfoLog);
				fprintf(gpFile, "\n--------------------------------------------------- \nCompile time log for Vertex Shader : \n %s", szInfoLog);
				fflush(gpFile);
				free(szInfoLog);
				return(-6);
			}
		}
	}
	// vertex shader part end
#pragma endregion

#pragma region FRAGMENT_SHADER
	// fragment shader part start

	fprintf(gpFile, "IN : initialize() FOR FRAGMENT SHADER \n");
	fflush(gpFile);

	// define fragment shader object
	gFragmentShaderObject = glCreateShader(GL_FRAGMENT_SHADER);

	fprintf(gpFile, "IN : initialize() done upto glCreateShader() \n");
	fflush(gpFile);

	// write fragment shader code
	const GLchar* fragmentShaderSourceCode =
		"#version 450 core" \
		"\n" \
		"in vec2 out_texcoord;" \
		"out vec4 FragColor;" \
		"uniform sampler2D u_sampler;" \
		"void main(void)" \
		"{" \
			"FragColor = texture(u_sampler,out_texcoord);" \
		"}";

	// specify above source, to fragment shader object
	glShaderSource(gFragmentShaderObject, 1, (GLchar**)&fragmentShaderSourceCode, NULL);

	fprintf(gpFile, "IN : initialize() done upto glShaderSource() \n");
	fflush(gpFile);

	// compile the fragment shader
	glCompileShader(gFragmentShaderObject);

	fprintf(gpFile, "IN : initialize() done upto glCompileShader() \n");
	fflush(gpFile);

	// error checking for fragment shader
	iShaderCompileStatus = 0;
	iInfoLogLength = 0;
	szInfoLog = NULL;

	glGetShaderiv(gFragmentShaderObject, GL_COMPILE_STATUS, &iShaderCompileStatus);

	fprintf(gpFile, "IN : initialize() done upto 1st glGetShaderiv() \n");
	fflush(gpFile);

	if (iShaderCompileStatus == GL_FALSE)
	{
		glGetShaderiv(gFragmentShaderObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0)
		{
			szInfoLog = (GLchar*)malloc(iInfoLogLength);
			if (szInfoLog != NULL)
			{
				GLsizei written;
				glGetShaderInfoLog(gFragmentShaderObject, iInfoLogLength, &written, szInfoLog);
				fprintf(gpFile, "\n--------------------------------------------------- \nCompile time log for Fragment Shader : \n %s", szInfoLog);
				fflush(gpFile);
				free(szInfoLog);
				return(-7);
			}
		}
	}
	// fragment shader part end
#pragma endregion

#pragma region SHADER_PROGRAM

	fprintf(gpFile, "IN : initialize() FOR SHADER PROGRAM \n");
	fflush(gpFile);

	// create shader program object
	gShaderProgramObject = glCreateProgram();

	fprintf(gpFile, "IN : initialize() done upto glCreateProgram() \n");
	fflush(gpFile);

	// attach vertex shader to shader program
	glAttachShader(gShaderProgramObject, gVertexShaderObject);

	fprintf(gpFile, "IN : initialize() done upto glAttachShader() for vertex shader \n");
	fflush(gpFile);

	// attach fragment shader to shader program
	glAttachShader(gShaderProgramObject, gFragmentShaderObject);

	fprintf(gpFile, "IN : initialize() done upto glAttachShader() for fragment shader \n");
	fflush(gpFile);


	// prelinking, binding to vertex attribute
	glBindAttribLocation(gShaderProgramObject, AMC_ATTRIBUTE_POSITION, "vPosition");
	glBindAttribLocation(gShaderProgramObject, AMC_ATTRIBUTE_TEXCOORD0, "vTexcoord");

	fprintf(gpFile, "IN : initialize() done upto glBindAttribLocation() \n");
	fflush(gpFile);

	// link the shader program
	glLinkProgram(gShaderProgramObject);

	fprintf(gpFile, "IN : initialize() done upto glLinkProgram() \n");
	fflush(gpFile);

	// error checking for shader program
	iProgramLinkStatus = 0;
	iInfoLogLength = 0;
	szInfoLog = NULL;

	glGetProgramiv(gShaderProgramObject, GL_LINK_STATUS, &iProgramLinkStatus);

	fprintf(gpFile, "IN : initialize() done upto 1st glGetProgramiv() \n");
	fflush(gpFile);

	if (iProgramLinkStatus == GL_FALSE)
	{
		glGetProgramiv(gShaderProgramObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0)
		{
			szInfoLog = (GLchar*)malloc(iInfoLogLength);
			if (szInfoLog != NULL)
			{
				GLsizei written;
				glGetProgramInfoLog(gShaderProgramObject, iInfoLogLength, &written, szInfoLog);
				fprintf(gpFile, "\n--------------------------------------------------- \nLink time log for Shader Program : \n %s", szInfoLog);
				fflush(gpFile);
				free(szInfoLog);
				return(-8);
			}
		}
	}


	// postlinking, getting uniform location
	mvpUniform = glGetUniformLocation(gShaderProgramObject, "u_mvp_matrix");
	samplerUniform = glGetUniformLocation(gShaderProgramObject, "u_sampler");
#pragma endregion


#pragma region RECTANGLE

	/*const GLfloat rectangleVertices[] =
	{
		1.0f, 1.0f,0.0f,
		-1.0f, 1.0f,0.0f,
		-1.0f, -1.0f,0.0f,
		1.0f, -1.0f,0.0f
	};*/

	const GLfloat rectangleTexcoords[] =
	{
		1.0f, 1.0f,
		0.0f, 1.0f,
		0.0f, 0.0f,
		1.0f, 0.0f
	};

	// create vao
	glGenVertexArrays(1, &vao_rectangle);
	glBindVertexArray(vao_rectangle);

	// create vbo for position
	glGenBuffers(1, &vbo_position_rectangle);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_position_rectangle);
	glBufferData(GL_ARRAY_BUFFER, 4 * 2 * sizeof(GLfloat), NULL, GL_DYNAMIC_DRAW);
	glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// create vbo for texture
	glGenBuffers(1, &vbo_texture_rectangle);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_texture_rectangle);
	glBufferData(GL_ARRAY_BUFFER, sizeof(rectangleTexcoords), rectangleTexcoords, GL_DYNAMIC_DRAW);
	glVertexAttribPointer(AMC_ATTRIBUTE_TEXCOORD0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(AMC_ATTRIBUTE_TEXCOORD0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);

#pragma endregion

	// for black screen
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	// for depth
	glClearDepth(1.0f);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);

	PerspectiveProjectionMatrix = mat4::identity();

	glEnable(GL_TEXTURE_2D);
	
	loadTexture();

	resize(WIN_WIDTH, WIN_HEIGHT);			// this leads to upper part cut

	fprintf(gpFile, "OUT : initialize() \n");
	fflush(gpFile);

	return(0);
}

void loadTexture(void)
{
	void InitializeGPU(int, unsigned char*);

	// code

	// CUDA
	InitializeGPU(CHECK_IMAGE_WIDTH, CheckImage);

	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

	glGenTextures(1, &texImage);

	glBindTexture(GL_TEXTURE_2D, texImage);

	// wrapping around S and T (repeted manner)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, CHECK_IMAGE_WIDTH, CHECK_IMAGE_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, CheckImage);

	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
}

void resize(int iWidth, int iHeight)
{
	fprintf(gpFile, "IN : resize() \n");
	fflush(gpFile);

	if (iHeight == 0)
	{
		iHeight = 1;
	}

	glViewport(0, 0, (GLsizei)iWidth, (GLsizei)iHeight);

	PerspectiveProjectionMatrix = perspective(45.0f, ((GLfloat)iWidth / (GLfloat)iHeight), 0.1f, 100.0f);

	fprintf(gpFile, "OUT : resize() \n");
	fflush(gpFile);
}

void display(void)
{
	
	// variable declaration
	GLfloat rectangleVertices[] =
	{
		-1.0f, -1.0f, 0.0f,
		-1.0f, 1.0f, 0.0f,
		1.0f, 1.0f, 0.0f,
		1.0f, -1.0f, 0.0f
	};

	// code
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glUseProgram(gShaderProgramObject);

	// declaration of matrices
	mat4 translationMatrix;
	mat4 rotatationMatrix;
	mat4 modelViewMatrix;
	mat4 modelViewProjectionMatrix;

	// .................................................................

	// RECTANGLE

	// matrices to identity
	translationMatrix = mat4::identity();
	rotatationMatrix = mat4::identity();
	modelViewMatrix = mat4::identity();
	modelViewProjectionMatrix = mat4::identity();


	// do necessary transformations
	translationMatrix = translate(0.0f, 0.0f, -1.0f);
	rotatationMatrix = rotate(0.0f, 0.0f, -90.0f);

	// do necessary matrix multiplications
	modelViewMatrix = translationMatrix;
	modelViewMatrix = modelViewMatrix * rotatationMatrix;
	modelViewProjectionMatrix = PerspectiveProjectionMatrix * modelViewMatrix;

	// send necessary matrices to shader in respective uniform
	glUniformMatrix4fv(mvpUniform, 1, GL_FALSE, modelViewProjectionMatrix);

	// ABU for texture
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, texImage);
	glUniform1i(GL_TEXTURE_2D, 0);

	// bind with vao
	glBindVertexArray(vao_rectangle);

	glBindBuffer(GL_ARRAY_BUFFER, vbo_position_rectangle);
	glBufferData(GL_ARRAY_BUFFER, sizeof(rectangleVertices), rectangleVertices, GL_DYNAMIC_DRAW);
	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	// unbind vao
	glBindVertexArray(0);

	// unuse program
	glUseProgram(0);

	SwapBuffers(ghdc);	// for double buffer
}

void update(void)
{
	

	
}

void uninitialize(void)
{

	fprintf(gpFile, "IN : uninitialize() \n");
	fflush(gpFile);

	GLsizei ShaderCount;
	GLsizei ShaderNumber;

	if (bIsFullScreen == true)
	{
		SetWindowLong(ghwnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(ghwnd, &wpPrev);
		SetWindowPos(ghwnd,
			HWND_TOP,
			0,
			0,
			0,
			0,
			SWP_NOZORDER | SWP_FRAMECHANGED | SWP_NOMOVE | SWP_NOSIZE | SWP_NOOWNERZORDER);
		ShowCursor(TRUE);
	}

	if (vbo_position_rectangle)
	{
		glDeleteBuffers(1, &vbo_position_rectangle);
		vbo_position_rectangle = 0;
	}

	if (vbo_texture_rectangle)
	{
		glDeleteBuffers(1, &vbo_texture_rectangle);
		vbo_texture_rectangle = 0;
	}

	if (vao_rectangle)
	{
		glDeleteVertexArrays(1, &vao_rectangle);
		vao_rectangle = 0;
	}

	if (texImage)
	{
		glBindTexture(GL_TEXTURE_2D, 0);
		texImage = 0;
	}


	if (gShaderProgramObject)
	{
		glUseProgram(gShaderProgramObject);
		glGetProgramiv(gShaderProgramObject, GL_ATTACHED_SHADERS, &ShaderCount);
		GLuint* pShaders = (GLuint*)malloc(sizeof(GLuint) * ShaderCount);
		if (pShaders)
		{
			glGetAttachedShaders(gShaderProgramObject, ShaderCount, &ShaderCount, pShaders);
			for (ShaderNumber = 0; ShaderNumber < ShaderCount; ShaderNumber++)
			{
				glDetachShader(gShaderProgramObject, pShaders[ShaderNumber]);
				glDeleteShader(pShaders[ShaderNumber]);
				pShaders[ShaderNumber] = 0;
			}
			free(pShaders);
		}
		glDeleteProgram(gShaderProgramObject);
		gShaderProgramObject = 0;
		glUseProgram(0);
	}


	if (wglGetCurrentContext() == ghrc)
	{
		wglMakeCurrent(NULL, NULL);
	}

	if (ghrc)
	{
		wglDeleteContext(ghrc);
		ghrc = NULL;
	}

	if (ghdc)
	{
		ReleaseDC(ghwnd, ghdc);
		ghdc = NULL;
	}

	if (gpFile)
	{
		fprintf(gpFile, "Log File is Closed !!\n\n");
		fflush(gpFile);
		fclose(gpFile);
		gpFile = NULL;
	}

}
