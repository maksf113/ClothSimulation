#pragma once
#include <vector>
#include <array>
#include <memory>
#include <GLFW/glfw3.h>
#include <imgui_impl_glfw.h>

#include "InputRecievers.hpp"

class InputManager
{
public:
	InputManager(GLFWwindow* window);
	~InputManager() = default;
	void handleKeyEvents(int key, int action, int scancode, int mods);
	void handleCurosrPosEvents(double x, double y);
	void handleMouseButtonEvents(int button, int action, int mods);
	void handleScrollEvents(double xoffset, double yoffset);
	void addKeyReciever(std::shared_ptr<KeyReciever> keyReciever);
	void addCursorPosReciever(std::shared_ptr<CursorPosReciever> keyReciever);
	bool isKeyPressed(int key) const;
	double cursorDX() const;
	double cursorDY() const;
	void endFrame();
private:
	void setKeyState(int key, int action);
private:
	GLFWwindow* m_window;
	std::array<bool, GLFW_KEY_LAST + 1> m_keyStates{false};
	double m_cursorLastX;
	double m_cursorLastY;
	double m_cursorDX = 0.0;
	double m_cursorDY = 0.0;
	bool m_firstCursorMove = true;
	std::vector<std::shared_ptr<KeyReciever>> m_keyRecievers;
	std::vector<std::shared_ptr<CursorPosReciever>> m_cursorPosRecievers;
};

inline InputManager::InputManager(GLFWwindow* window) : m_window(window) {}

inline void InputManager::handleKeyEvents(int key, int scancode, int action, int mods)
{
	ImGui_ImplGlfw_KeyCallback(m_window, key, scancode, action, mods);
	ImGuiIO& io = ImGui::GetIO();
	if (io.WantCaptureKeyboard)
		return;
	setKeyState(key, action);
	if(action == GLFW_PRESS)
		for (auto& keyReciever : m_keyRecievers)
			keyReciever->handleKeyEvents(key, scancode, action, mods);
}

inline void InputManager::handleCurosrPosEvents(double x, double y)
{
	ImGuiIO& io = ImGui::GetIO();
	if (io.WantCaptureMouse)
	{
		m_firstCursorMove = true;
		return;
	}
	if (m_firstCursorMove)
	{
		m_cursorLastX = x;
		m_cursorLastY = y;
		m_firstCursorMove = false;
	}
	m_cursorDX = x - m_cursorLastX;
	m_cursorDY = m_cursorLastY - y;
	m_cursorLastX = x;
	m_cursorLastY = y;
	for (auto& curosorPosReciever : m_cursorPosRecievers)
		curosorPosReciever->handleCursorPosEvents(x, y);
}

inline void InputManager::handleMouseButtonEvents(int button, int action, int mods)
{
	ImGui_ImplGlfw_MouseButtonCallback(m_window, button, action, mods);
	ImGuiIO& io = ImGui::GetIO();
	if (io.WantCaptureMouse)
		return;
}

inline void InputManager::handleScrollEvents(double xoffset, double yoffset)
{
	ImGui_ImplGlfw_ScrollCallback(m_window, xoffset, yoffset);
	ImGuiIO& io = ImGui::GetIO();
	if (io.WantCaptureMouse)
		return;
}

inline void InputManager::addKeyReciever(std::shared_ptr<KeyReciever> keyReciever)
{
	m_keyRecievers.push_back(keyReciever);
}

inline void InputManager::addCursorPosReciever(std::shared_ptr<CursorPosReciever> crsorPosReciever)
{
	m_cursorPosRecievers.push_back(crsorPosReciever);
}

inline bool InputManager::isKeyPressed(int key) const
{
	if (key >= 0 && key < m_keyStates.size())
		return m_keyStates[key];
	else
		return false;
}

inline double InputManager::cursorDX() const
{
	return m_cursorDX;
}

inline double InputManager::cursorDY() const
{
	return m_cursorDY;
}

inline void InputManager::endFrame()
{
	m_cursorDX = 0.0;
	m_cursorDY = 0.0;
}

inline void InputManager::setKeyState(int key, int action)
{
	if (key >= 0 && key < m_keyStates.size())
	{
		if (action == GLFW_PRESS || action == GLFW_REPEAT)
		{
			m_keyStates[key] = true;
		}
		else if (action == GLFW_RELEASE)
		{
			m_keyStates[key] = false;
		}
	}
}

