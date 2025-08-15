#pragma once
#ifndef GLM_ENABLE_EXPERIMENTAL
#define GLM_ENABLE_EXPERIMENTAL
#endif
#include<glm/glm.hpp>
#include<glm/gtc/matrix_transform.hpp>
#include<glm/gtc/type_ptr.hpp>

#include "tools/InputManager.hpp"

class Camera
{
public:
	Camera();
	~Camera() = default;
	glm::mat4 view() const;
	glm::mat4 projection(float aspect) const;
	void processInput(const InputManager& im, double dt, bool cursorEnabled);
private:
	glm::vec3 m_position = glm::vec3(0.0f, 1.0f, -4.0f);
	glm::vec3 m_forward = glm::vec3(0.0f, 0.0f, 1.0f);
	glm::vec3 m_right = glm::vec3(0.0f, 0.0f, 0.0f);
	glm::vec3 m_up = glm::vec3(0.0f, 1.0f, 0.0f);
	glm::vec3 m_worldUp = glm::vec3(0.0f, 1.0f, 0.0f);
	float m_azimuth = 90.0f;
	float m_elevation = 0.0f;
	float m_fov = 45.0f;
	float m_near = 0.05f;
	float m_far = 100.0f;
	float m_speed = 0.002f;
	float m_mouseSensitivity = 0.1f;
};

inline Camera::Camera()
{
	float cosAzim = std::cosf(glm::radians(m_azimuth));
	float sinAzim = std::sinf(glm::radians(m_azimuth));
	float cosElev = std::cosf(glm::radians(m_elevation));
	float sinElev = std::sinf(glm::radians(m_elevation));
	m_forward = glm::vec3(cosAzim * cosElev, sinElev, sinAzim * cosElev);
	m_right = glm::normalize(glm::cross(m_forward, m_up));
}

glm::mat4 Camera::view() const
{
	return glm::lookAt(m_position, m_position + m_forward, m_worldUp);
}

inline glm::mat4 Camera::projection(float aspect) const
{
	return glm::perspective(glm::radians(m_fov), aspect, m_near, m_far);
}

inline void Camera::processInput(const InputManager& im, double dt, bool cursorEnabled)
{
	if (cursorEnabled)
		return;
	// --- keyboard ---
	// project forward onto x-z plane
	glm::vec3 forward = glm::normalize(glm::vec3(m_forward.x, 0.0f, m_forward.z));
	// distance increment
	float dx = m_speed * dt;
	if (im.isKeyPressed(GLFW_KEY_W))
		m_position += forward * dx;
	if (im.isKeyPressed(GLFW_KEY_S))
		m_position -= forward * dx;
	if (im.isKeyPressed(GLFW_KEY_A))
		m_position -= m_right * dx;
	if (im.isKeyPressed(GLFW_KEY_D))
		m_position += m_right * dx;
	if (im.isKeyPressed(GLFW_KEY_Q))
		m_position -= m_up * dx;
	if (im.isKeyPressed(GLFW_KEY_E))
		m_position += m_up * dx;
	// --- mouse ---
	m_azimuth += m_mouseSensitivity * im.cursorDX();
	m_elevation += m_mouseSensitivity * im.cursorDY();
	// check bounds
	if (m_elevation > 89.0f)
		m_elevation = 89.0f;
	if (m_elevation < -89.0f)
		m_elevation = -89.0f;
	if (m_azimuth >= 360.0f)
		m_azimuth -= 360.0f;
	if (m_azimuth < 0.0f)
		m_azimuth += 360.0f;
	float cosAzim = std::cosf(glm::radians(m_azimuth));
	float sinAzim = std::sinf(glm::radians(m_azimuth));
	float cosElev = std::cosf(glm::radians(m_elevation));
	float sinElev = std::sinf(glm::radians(m_elevation));
	m_forward = glm::vec3(cosAzim * cosElev, sinElev, sinAzim * cosElev);
	m_right = glm::normalize(glm::cross(m_forward, m_up));
}

