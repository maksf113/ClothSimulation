#pragma once
#include <chrono>

class Timer
{
public:
	Timer() = default;
	~Timer() = default;
	void start();
	void stop();
	double elapsedSeconds() const;
	double elapsedMiliseconds() const;
private:
	std::chrono::time_point<std::chrono::high_resolution_clock> m_startTime;
	std::chrono::time_point<std::chrono::high_resolution_clock> m_endTime;
	bool m_isRunning = false;
};

inline void Timer::start()
{
	m_startTime = std::chrono::high_resolution_clock::now();
	m_isRunning = true;
}

inline void Timer::stop()
{
	if (!m_isRunning)
		return;
	m_endTime = std::chrono::high_resolution_clock::now();
	m_isRunning = false;
}

inline double Timer::elapsedSeconds() const
{
	auto endTime = m_isRunning ? std::chrono::high_resolution_clock::now() : m_endTime;
	return std::chrono::duration<double>(endTime - m_startTime).count();
}

inline double Timer::elapsedMiliseconds() const
{
	auto endTime = m_isRunning ? std::chrono::high_resolution_clock::now() : m_endTime;
	return std::chrono::duration<double, std::milli>(endTime - m_startTime).count();
}

