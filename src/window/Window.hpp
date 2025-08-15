#pragma once
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <stdexcept>

#include "tools/InputRecievers.hpp"
#include "tools/InputManager.hpp"
#include "graphics/Renderer.hpp"


class Window : public KeyReciever
{
public:
	Window(uint32_t width, uint32_t height);
	~Window();
	GLFWwindow* windowPtr();
	uint32_t width() const;
	uint32_t height() const;
	void setTitle(const char* title);
	void swapBuffers();
	void pollEvents();
	bool shouldClose() const;
	void setCallbacks(InputManager* im);
	bool isCursorEnabled() const;
private:
	void handleKeyEvents(int key, int action, int scancode, int mods);
private:
	GLFWwindow* m_handle;
	uint32_t m_width;
	uint32_t m_height;
	bool m_cursorEnabled = false;
}; 

Window::Window(uint32_t width, uint32_t height) : m_width(width), m_height(height)
{
	if (!glfwInit())
		throw std::runtime_error("GLFW initialization failed");
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);

	m_handle = glfwCreateWindow(width, height, "Cloth Simulation", nullptr, nullptr);
	if (!m_handle)
		throw std::runtime_error("Window creation failure");
	glfwMakeContextCurrent(m_handle);

	if (glewInit() != GLEW_OK)
	{
		throw std::runtime_error("GLEW initialization failed");
		glfwDestroyWindow(m_handle);
		glfwTerminate();
	}

	glfwSwapInterval(1);

	glfwSetInputMode(m_handle, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	if (glfwRawMouseMotionSupported())
		glfwSetInputMode(m_handle, GLFW_CURSOR, GLFW_RAW_MOUSE_MOTION);
}
Window::~Window()
{
	glfwDestroyWindow(m_handle);
	glfwTerminate();
}
inline GLFWwindow* Window::windowPtr()
{
	return m_handle; 
}
inline uint32_t Window::width() const
{
	return m_width;
}
inline uint32_t Window::height() const
{
	return m_height;
}
inline void Window::setTitle(const char* title)
{
	glfwSetWindowTitle(m_handle, title);
}
inline void Window::swapBuffers()
{
	glfwSwapBuffers(m_handle);
}
inline void Window::pollEvents()
{
	glfwPollEvents();
}
inline bool Window::shouldClose() const
{
	return glfwWindowShouldClose(m_handle);
}

inline void Window::handleKeyEvents(int key, int action, int scancode, int mods)
{
	if (glfwGetKey(m_handle, GLFW_KEY_ESCAPE) == GLFW_PRESS)
	{
		m_cursorEnabled = !m_cursorEnabled;
		if (m_cursorEnabled)
		{
			glfwSetInputMode(m_handle, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
		}
		else
		{
			glfwSetInputMode(m_handle, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
			if (glfwRawMouseMotionSupported())
				glfwSetInputMode(m_handle, GLFW_CURSOR, GLFW_RAW_MOUSE_MOTION);
		}
	}
}

inline void Window::setCallbacks(InputManager* im)
{
	glfwSetWindowUserPointer(m_handle, im);
	glfwSetKeyCallback(m_handle, [](GLFWwindow* win, int key, int scancode, int action, int mods)
		{
			InputManager* im = static_cast<InputManager*>(glfwGetWindowUserPointer(win));
			im->handleKeyEvents(key, scancode, action, mods);
		});
	glfwSetCursorPosCallback(m_handle, [](GLFWwindow* win, double x, double y) 
		{
			InputManager* im = static_cast<InputManager*>(glfwGetWindowUserPointer(win));
			im->handleCurosrPosEvents(x, y);
		});
	glfwSetMouseButtonCallback(m_handle, [](GLFWwindow* win, int button, int action, int mods)
		{
			InputManager* im = static_cast<InputManager*>(glfwGetWindowUserPointer(win));
			im->handleMouseButtonEvents(button, action, mods);
		});
	glfwSetScrollCallback(m_handle, [](GLFWwindow* win, double xoffset, double yoffset)
		{
			InputManager* im = static_cast<InputManager*>(glfwGetWindowUserPointer(win));
			im->handleScrollEvents(xoffset, yoffset);
		});
}

inline bool Window::isCursorEnabled() const
{
	return m_cursorEnabled;
}
