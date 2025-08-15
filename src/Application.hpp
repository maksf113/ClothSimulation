#pragma once
#include <memory>

#include "window/Window.hpp"

#include "tools/GUI.hpp"
#include "tools/InputManager.hpp"
#include "tools/Timer.hpp"

#include "graphics/Renderer.hpp"

#include "cuda/CudaProcessor.hpp"

class Application
{
public:
	Application();
	~Application() = default;
	void run();
private:
	std::shared_ptr<Window> m_window;
	std::shared_ptr<Renderer> m_renderer;
	std::unique_ptr<InputManager> m_inputManager;
	std::unique_ptr<GUI> m_gui;
	Timer m_timer;
	double m_lastTime = 0.0;
	bool m_isRunning = true;
};

Application::Application()
{
	constexpr uint32_t width = 960;
	constexpr uint32_t height = 640;
	m_window = std::make_shared<Window>(width, height);
	m_renderer = std::make_shared<Renderer>(width, height);
	m_inputManager = std::make_unique<InputManager>(m_window->windowPtr());
	m_gui = std::make_unique<GUI>(m_window->windowPtr());
	m_window->setCallbacks(m_inputManager.get());
	m_inputManager->addKeyReciever(m_window);
}

inline void Application::run()
{
	m_timer.start();
	while (m_isRunning)
	{
		double currentTime = m_timer.elapsedMiliseconds();
		double dt = currentTime - m_lastTime;
		m_lastTime = currentTime;
		m_renderer->processInput(*m_inputManager, dt, m_window->isCursorEnabled());
		m_renderer->draw(m_window->width(), m_window->height(), dt);
		m_inputManager->endFrame();
		m_gui->createFrame(m_renderer->getClothParams(), m_renderer->simulate());
		m_gui->draw();
		m_window->pollEvents();
		m_window->swapBuffers();
		if (m_window->shouldClose())
			m_isRunning = false;
	}
	m_timer.stop();
}
